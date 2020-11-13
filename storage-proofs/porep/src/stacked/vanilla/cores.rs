use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, RwLock};

use anyhow::Result;
use hwloc2::{CpuBindFlags, ObjectType, Topology, TopologyObject};
use lazy_static::lazy_static;
use log::*;

use storage_proofs_core::settings;

type CoreGroup = Vec<u32>;
lazy_static! {
    pub static ref TOPOLOGY: RwLock<Topology> =
        RwLock::new(Topology::new().expect("unable to load topology"));
    pub static ref CORE_GROUPS: (Vec<Mutex<CoreGroup>>, usize) = {
        let settings = &settings::SETTINGS;
        let num_producers = settings.multicore_sdr_producers;
        let cores_per_unit = num_producers + 1;

        core_groups(cores_per_unit)
    };
}

pub fn actual_cores_per_unit() -> usize {
    CORE_GROUPS.1
}

pub fn checkout_core_group() -> Option<MutexGuard<'static, CoreGroup>> {
    for (i, group) in CORE_GROUPS.0.iter().enumerate() {
        match group.try_lock() {
            Ok(guard) => {
                debug!("checked out core group {}", i);
                return Some(guard);
            }
            Err(_) => debug!("core group {} locked, could not checkout", i),
        }
    }
    None
}

#[cfg(not(target_os = "windows"))]
pub type ThreadId = libc::pthread_t;

#[cfg(target_os = "windows")]
pub type ThreadId = winapi::winnt::HANDLE;

/// Helper method to get the thread id through libc, with current rust stable (1.5.0) its not
/// possible otherwise I think.
#[cfg(not(target_os = "windows"))]
fn get_thread_id() -> ThreadId {
    unsafe { libc::pthread_self() }
}

#[cfg(target_os = "windows")]
fn get_thread_id() -> ThreadId {
    unsafe { kernel32::GetCurrentThread() }
}

pub struct Cleanup {
    tid: ThreadId,
    prior_state: Option<hwloc2::Bitmap>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        if let Some(prior) = self.prior_state.take() {
            if let Err(err) = TOPOLOGY.write().unwrap().set_cpubind_for_thread(
                self.tid,
                prior,
                CpuBindFlags::CPUBIND_THREAD,
            ) {
                warn!(
                    "failed to remove cpubinding for thread {}: {:?}",
                    self.tid, err
                );
            }
        }
    }
}

pub fn bind_core(logical_index: u32) -> Result<Cleanup> {
    let child_topo = &mut *TOPOLOGY.write().unwrap();
    let tid = get_thread_id();
    let core = get_core_by_index(&child_topo, logical_index).map_err(|err| {
        anyhow::format_err!(
            "failed to get core at logical index {}: {:?}",
            logical_index,
            err
        )
    })?;

    let cpuset = core.cpuset().ok_or_else(|| {
        anyhow::format_err!(
            "no allowed cpuset for core at logical index {}",
            logical_index
        )
    })?;
    debug!("allowed cpuset: {:?}", cpuset);
    let mut bind_to = cpuset;

    // Get only one logical processor (in case the core is SMT/hyper-threaded).
    bind_to.singlify();

    // Thread binding before explicit set.
    let before = child_topo.get_cpubind_for_thread(tid, CpuBindFlags::CPUBIND_THREAD);

    debug!("binding to {:?}", bind_to);
    // Set the binding.
    let result = child_topo
        .set_cpubind_for_thread(tid, bind_to, CpuBindFlags::CPUBIND_THREAD)
        .map_err(|err| anyhow::format_err!("failed to bind CPU: {:?}", err));

    if result.is_err() {
        warn!("error in bind_core, {:?}", result);
    }

    Ok(Cleanup {
        tid,
        prior_state: before,
    })
}

fn get_core_by_index(topo: &Topology, index: u32) -> Result<&TopologyObject> {
    match topo.objects_with_type(&ObjectType::Core) {
        Ok(all_cores) => all_cores
            .iter()
            .find(|core| core.logical_index() == index)
            .map(|core| *core)
            .ok_or_else(|| anyhow::format_err!("failed to get core by logical index {}", index)),
        _e => Err(anyhow::format_err!("failed to get core by index {}", index)),
    }
}

fn core_groups(cores_per_unit: usize) -> (Vec<Mutex<CoreGroup>>, usize) {
    let topo = &*TOPOLOGY.read().unwrap();

    let all_cores = topo.objects_with_type(&ObjectType::Core).unwrap();

    let mut groups = HashMap::new();
    for core in &all_cores {
        let mut parent = core.parent();

        while let Some(p) = parent {
            match p.object_type() {
                ObjectType::L3Cache => {
                    let list = groups.entry(p.logical_index()).or_insert(Vec::new());
                    list.push(core);
                    break;
                }
                _ => {}
            }

            parent = p.parent();
        }
    }

    let group_sizes = groups.values().map(|group| group.len()).collect::<Vec<_>>();
    let group_count = groups.len();

    debug!(
        "found {} shared cache group(s), with sizes {:?}",
        group_count, group_sizes
    );

    // assumes that all groups are the same size
    // if the shared L3 cache group is larger, or only 1, do manual grouping.
    if group_sizes[0] >= 2 * cores_per_unit || group_sizes[0] == 1 {
        let groups = groups
            .into_iter()
            .flat_map(|(_, group)| {
                group
                    .chunks(cores_per_unit)
                    .map(|cores| {
                        Mutex::new(
                            cores
                                .iter()
                                .map(|core| core.logical_index())
                                .collect::<Vec<u32>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        (groups, cores_per_unit)
    } else {
        let groups = groups
            .into_iter()
            .map(|(_, group)| {
                Mutex::new(group.into_iter().map(|core| core.logical_index()).collect())
            })
            .collect::<Vec<_>>();
        (groups, group_sizes[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cores() {
        core_groups(2);
    }

    #[test]
    fn test_checkout_cores() {
        let checkout1 = checkout_core_group();
        dbg!(&checkout1);
        let checkout2 = checkout_core_group();
        dbg!(&checkout2);

        // This test might fail if run on a machine with fewer than four cores.
        match (checkout1, checkout2) {
            (Some(c1), Some(c2)) => assert!(*c1 != *c2),
            _ => panic!("failed to get two checkouts"),
        }
    }
}

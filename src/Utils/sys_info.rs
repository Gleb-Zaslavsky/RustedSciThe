/*
A tool for pretty printing system information about the computer. Used in addition to performance information.
*/
use tabled::settings::Style;
use sys_info;
use tabled::{Table, Tabled};
#[derive(PartialEq)]
#[derive(Tabled)]
pub struct SystemInfo {
    key: &'static str,
    value: String,
}
#[allow(dead_code)]
pub fn this_system_info() ->Vec< SystemInfo> {
    // Gather system information
    let cpu_num = sys_info::cpu_num().unwrap_or_default();
    let cpu_speed = sys_info::cpu_speed().unwrap_or_default();
    let disk_info = sys_info::disk_info().unwrap();
    let loadavg = sys_info::loadavg().unwrap();
    let mem_info = sys_info::mem_info().unwrap();
    let os_type = sys_info::os_type().unwrap_or_default();
    let os_release = sys_info::os_release().unwrap_or_default();
    let proc_total = sys_info::proc_total().unwrap_or_default();

    // Create a vector of SystemInfo structs
    let system_info = vec![
        SystemInfo { key: "CPU Cores", value: cpu_num.to_string() },
        SystemInfo { key: "CPU Speed (MHz)", value: cpu_speed.to_string() },
        SystemInfo { key: "Disk Total (KB)", value: disk_info.total.to_string() },
        SystemInfo { key: "Disk Free (KB)", value: disk_info.free.to_string() },
        SystemInfo { key: "Load Average (1 min)", value: loadavg.one.to_string() },
        SystemInfo { key: "Load Average (5 min)", value: loadavg.five.to_string() },
        SystemInfo { key: "Load Average (15 min)", value: loadavg.fifteen.to_string() },
        SystemInfo { key: "Memory Total (KB)", value: mem_info.total.to_string() },
        SystemInfo { key: "Memory Free (KB)", value: mem_info.free.to_string() },
        SystemInfo { key: "OS Type", value: os_type },
        SystemInfo { key: "OS Release", value: os_release },
        SystemInfo { key: "Total Processes", value: proc_total.to_string() },
    ];

    // Create a table from the system_info vector
    let mut table = Table::new(&system_info);
    table.with(Style::modern_rounded());

    // Print the table
    println!("System Information\n\n");
    println!("{}", table);
    system_info
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
fn test_system_info_contains_keys() {
    let output = this_system_info();

    // Check that the output contains expected keys
    assert!(output.iter().any(|info| info.key == "CPU Cores"));
    assert!(output.iter().any(|info| info.key == "CPU Speed (MHz)"));
    assert!(output.iter().any(|info| info.key == "Disk Total (KB)"));
    assert!(output.iter().any(|info| info.key == "Disk Free (KB)"));
    assert!(output.iter().any(|info| info.key == "Load Average (1 min)"));
    assert!(output.iter().any(|info| info.key == "Load Average (5 min)"));
    assert!(output.iter().any(|info| info.key == "Load Average (15 min)"));
    assert!(output.iter().any(|info| info.key == "Memory Total (KB)"));
    assert!(output.iter().any(|info| info.key == "Memory Free (KB)"));
    assert!(output.iter().any(|info| info.key == "OS Type"));
    assert!(output.iter().any(|info| info.key == "OS Release"));
    assert!(output.iter().any(|info| info.key == "Total Processes"));
}
    #[test]
    fn test_system_info_not_empty() {
        let output = this_system_info();

        // Check that the output is not empty
        assert!(!output.is_empty());
    }
}

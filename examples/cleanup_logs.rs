use RustedSciThe::Utils::logger::delete_old_logs;

fn main() {
    // Delete old log files from the current directory
    match delete_old_logs(".") {
        Ok(()) => println!("Successfully cleaned up old log files"),
        Err(e) => eprintln!("Error cleaning up log files: {}", e),
    }
}

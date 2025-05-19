mod parser;

mod parse_task;
use parse_task::*;
pub fn utils_examples(example: usize) {
     match example {
        1 => {
            println!("Hello, world!");
            let input = r#"
                database
                connection_string: mongodb://user:pass@localhost:27017, postgresql://postgres:password@localhost:5432
                timeout: 30s, 1m, 5m
                
                paths
                log_dir: /var/log/app, C:\logs\app
                data_dir: /var/data, D:\data
                "#;

            let result = parse_document_as_strings(input, None).unwrap();
            println!("{:?}", result);
        }
        _ => {}
    }

}
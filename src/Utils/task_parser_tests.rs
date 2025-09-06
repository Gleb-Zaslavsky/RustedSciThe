/////////////////////////////TESTS////////////////////////////////////////////////////
/*
comprehensive tests:
Basic parsing test
Mixed type parsing test
Template usage test
Empty document test
Malformed document test
File-based parsing test
Complex document test
Special character handling test
*/

#[cfg(test)]
mod tests1 {

    use crate::Utils::task_parser::{
        Value, parse_document, parse_document_as_strings, parse_document_with_template, parse_key,
        parse_key_value_pair, parse_section, parse_this_sections, parse_title, parse_value,
        parse_value_list,
    };
    use std::collections::HashMap;
    #[test]
    fn test_parse_title() {
        // Basic title
        let (remaining, title) = parse_title("title1\n key1: value1").unwrap();
        assert_eq!(title, "title1");
        assert_eq!(remaining, "key1: value1");

        // Title with underscore
        let (remaining, title) = parse_title("title_with_underscore key1: value1").unwrap();
        assert_eq!(title, "title_with_underscore");
        assert_eq!(remaining, "key1: value1");

        // Title with numbers
        let (remaining, title) = parse_title("title123 key1: value1").unwrap();
        assert_eq!(title, "title123");
        assert_eq!(remaining, "key1: value1");
    }

    #[test]
    fn test_parse_key() {
        // Basic key
        let (remaining, key) = parse_key("key1: value1").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(remaining, ": value1");

        // Key with underscore
        let (remaining, key) = parse_key("key_with_underscore: value1").unwrap();
        assert_eq!(key, "key_with_underscore");
        assert_eq!(remaining, ": value1");

        // Key with numbers
        let (remaining, key) = parse_key("key123: value1").unwrap();
        assert_eq!(key, "key123");
        assert_eq!(remaining, ": value1");
    }

    #[test]
    fn test_parse_value() {
        // String value
        let (remaining, value) = parse_value("value1, value2").unwrap();
        assert_eq!(value, Value::String("value1".to_string()));
        assert_eq!(remaining, ", value2");

        // Integer value
        let (remaining, value) = parse_value("123, next").unwrap();
        assert_eq!(value, Value::Integer(123));
        assert_eq!(remaining, ", next");

        // Float value
        let (remaining, value) = parse_value("123.45, next").unwrap();
        assert_eq!(value, Value::Float(123.45));
        assert_eq!(remaining, ", next");

        // Boolean value
        let (remaining, value) = parse_value("true, next").unwrap();
        assert_eq!(value, Value::Boolean(true));
        assert_eq!(remaining, ", next");
    }

    #[test]
    fn test_parse_value_vectors() {
        // Vector of floats
        let (remaining, value) = parse_value("[1.0,2.5,3.14], next").unwrap();
        assert_eq!(value, Value::Vector(vec![1.0, 2.5, 3.14]));
        assert_eq!(remaining, ", next");

        // Empty vector
        let (remaining, value) = parse_value("[], next").unwrap();
        assert_eq!(value, Value::Vector(vec![]));
        assert_eq!(remaining, ", next");

        // Vector with spaces
        let (remaining, value) = parse_value("[1.0, 2.5, 3.14], next").unwrap();
        assert_eq!(value, Value::Vector(vec![1.0, 2.5, 3.14]));
        assert_eq!(remaining, ", next");
    }

    #[test]
    fn test_parse_value_options() {
        // None value
        let (remaining, value) = parse_value("None, next").unwrap();
        assert_eq!(value, Value::Optional(None));
        assert_eq!(remaining, ", next");

        // Some with integer
        let (remaining, value) = parse_value("Some(42), next").unwrap();
        assert_eq!(value, Value::Optional(Some(Box::new(Value::Integer(42)))));
        assert_eq!(remaining, ", next");

        // Some with float
        let (remaining, value) = parse_value("Some(3.14), next").unwrap();
        assert_eq!(value, Value::Optional(Some(Box::new(Value::Float(3.14)))));
        assert_eq!(remaining, ", next");

        // Some with string
        let (remaining, value) = parse_value("Some(hello), next").unwrap();
        assert_eq!(
            value,
            Value::Optional(Some(Box::new(Value::String("hello".to_string()))))
        );
        assert_eq!(remaining, ", next");

        // Some with boolean
        let (remaining, value) = parse_value("Some(true), next").unwrap();
        assert_eq!(value, Value::Optional(Some(Box::new(Value::Boolean(true)))));
        assert_eq!(remaining, ", next");
    }

    #[test]
    fn test_parse_value_list() {
        // Mixed type list
        let (remaining, values) = parse_value_list("value1, 123, 45.67, true").unwrap();
        assert_eq!(
            values,
            vec![
                Value::String("value1".to_string()),
                Value::Integer(123),
                Value::Float(45.67),
                Value::Boolean(true)
            ]
        );
        assert_eq!(remaining, "");

        // Single value
        let (remaining, values) = parse_value_list("value1").unwrap();
        assert_eq!(values, vec![Value::String("value1".to_string())]);
        assert_eq!(remaining, "");

        // Empty list
        let (remaining, values) = parse_value_list("").unwrap();
        assert_eq!(values, Vec::<Value>::new());
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_value_list_with_vectors_and_options() {
        // List with vectors and options
        let (remaining, values) =
            parse_value_list("[1.0,2.0], None, Some(42), Some(hello)").unwrap();
        assert_eq!(
            values,
            vec![
                Value::Vector(vec![1.0, 2.0]),
                Value::Optional(None),
                Value::Optional(Some(Box::new(Value::Integer(42)))),
                Value::Optional(Some(Box::new(Value::String("hello".to_string()))))
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_key_value_pair() {
        // Basic key-value pair with string values
        let (remaining, (key, values)) = parse_key_value_pair("key1: value1, value2").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(
            values,
            vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(remaining, "");

        // Key-value pair with mixed types
        let (remaining, (key, values)) =
            parse_key_value_pair("key1: value1, 123, 45.67, true").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(
            values,
            vec![
                Value::String("value1".to_string()),
                Value::Integer(123),
                Value::Float(45.67),
                Value::Boolean(true)
            ]
        );
        assert_eq!(remaining, "");

        // With trailing text
        let (remaining, (key, values)) = parse_key_value_pair("key1: value1, value2;").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(
            values,
            vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(remaining, ";");

        // With spaces
        let (remaining, (key, values)) = parse_key_value_pair("key1 : value1 , value2").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(
            values,
            vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(remaining, "");

        // Single value
        let (remaining, (key, values)) = parse_key_value_pair("key1: value1").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(values, vec![Value::String("value1".to_string())]);
        assert_eq!(remaining, "");

        // Empty value list
        let (remaining, (key, values)) = parse_key_value_pair("key1:").unwrap();
        assert_eq!(key, "key1");
        assert_eq!(values, Vec::<Value>::new());
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_section() {
        // Basic section with string values
        let input = "section1 key1: value1, value2 key2: value3, value4";
        let (remaining, (title, map)) = parse_section(input).unwrap();

        assert_eq!(title, "section1");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("key1").unwrap(),
            &vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(
            map.get("key2").unwrap(),
            &vec![
                Value::String("value3".to_string()),
                Value::String("value4".to_string())
            ]
        );
        assert_eq!(remaining, "");

        // Section with mixed value types
        let input = "section1 key1: value1, 123 key2: 45.67, true";
        let (remaining, (title, map)) = parse_section(input).unwrap();

        assert_eq!(title, "section1");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("key1").unwrap(),
            &vec![Value::String("value1".to_string()), Value::Integer(123)]
        );
        assert_eq!(
            map.get("key2").unwrap(),
            &vec![Value::Float(45.67), Value::Boolean(true)]
        );
        assert_eq!(remaining, "");

        // Section with trailing text
        let input = "section1 key1: value1, value2 key2: value3, value4 section2";
        let (remaining, (title, map)) = parse_section(input).unwrap();

        assert_eq!(title, "section1");
        assert_eq!(map.len(), 2);
        assert_eq!(remaining, "section2");

        // Section with single key
        let input = "section1 key1: value1, value2";
        let (remaining, (title, map)) = parse_section(input).unwrap();

        assert_eq!(title, "section1");
        assert_eq!(map.len(), 1);
        assert_eq!(
            map.get("key1").unwrap(),
            &vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_section_mixed_spacing() {
        // Section with varied spacing
        let input = "config  key1:value1,value2   key2 : value3 , value4";
        let (remaining, (title, map)) = parse_section(input).unwrap();

        assert_eq!(title, "config");
        assert_eq!(map.len(), 2);
        assert_eq!(
            map.get("key1").unwrap(),
            &vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(
            map.get("key2").unwrap(),
            &vec![
                Value::String("value3".to_string()),
                Value::String("value4".to_string())
            ]
        );
        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_document_basic() {
        // Basic document with two sections
        let input = "section1 key1: value1, value2 key2: value3\nsection2 key3: value4, value5";
        let doc = parse_document(input).unwrap();
        println!("\n {:?} \n", doc);
        assert_eq!(doc.len(), 2);

        // Check section1
        let section1 = &doc["section1"];
        assert_eq!(section1.len(), 2);
        assert_eq!(
            section1.get("key1").unwrap().as_ref().unwrap(),
            &vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(
            section1.get("key2").unwrap().as_ref().unwrap(),
            &vec![Value::String("value3".to_string())]
        );

        // Check section2
        let section2 = &doc["section2"];
        assert_eq!(section2.len(), 1);
        assert_eq!(
            section2.get("key3").unwrap().as_ref().unwrap(),
            &vec![
                Value::String("value4".to_string()),
                Value::String("value5".to_string())
            ]
        );
    }

    #[test]
    fn test_parse_document_with_mixed_types() {
        // Document with mixed value types
        let input = "section1 key1: value1, 123 key2:  45.67, true\nsection2 key3: false, 789";
        let doc = parse_document(input).unwrap();

        assert_eq!(doc.len(), 2);

        // Check section1
        let section1 = &doc["section1"];
        assert_eq!(section1.len(), 2);
        assert_eq!(
            section1.get("key1").unwrap().as_ref().unwrap(),
            &vec![Value::String("value1".to_string()), Value::Integer(123)]
        );
        assert_eq!(
            section1.get("key2").unwrap().as_ref().unwrap(),
            &vec![Value::Float(45.67), Value::Boolean(true)]
        );

        // Check section2
        let section2 = &doc["section2"];
        assert_eq!(section2.len(), 1);
        assert_eq!(
            section2.get("key3").unwrap().as_ref().unwrap(),
            &vec![Value::Boolean(false), Value::Integer(789)]
        );
    }

    #[test]
    fn test_parse_document_with_multiple_line_breaks() {
        // Document with multiple line breaks between sections
        let input = "section1 key1: value1, value2\n\n\nsection2 key2: value3, value4";
        let doc = parse_document(input).unwrap();

        assert_eq!(doc.len(), 2);
        assert!(doc.contains_key("section1"));
        assert!(doc.contains_key("section2"));
    }

    #[test]
    fn test_parse_document_empty() {
        // Empty document
        let input = "";

        assert_eq!(parse_document(input).is_err(), true);
    }

    #[test]
    fn test_parse_document_with_template() {
        // Create a template
        let mut template = HashMap::new();
        let mut section1_map = HashMap::new();
        section1_map.insert("key1".to_string(), None);
        section1_map.insert("key2".to_string(), None);
        section1_map.insert("key3".to_string(), None); // Not in the input

        let mut section2_map = HashMap::new();
        section2_map.insert("key4".to_string(), None);

        template.insert("section1".to_string(), section1_map);
        template.insert("section2".to_string(), section2_map);
        template.insert("section3".to_string(), HashMap::new()); // Not in the input

        // Parse with template
        let input = "section1 key1: value1, value2 key2: value3\nsection2 key4: value4, value5";
        let result = parse_document_with_template(input, &template).unwrap();

        // Check all sections exist
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("section1"));
        assert!(result.contains_key("section2"));
        assert!(result.contains_key("section3"));

        // Check section1
        let section1 = &result["section1"];
        assert_eq!(section1.len(), 3);
        assert_eq!(
            section1.get("key1").unwrap().as_ref().unwrap(),
            &vec![
                Value::String("value1".to_string()),
                Value::String("value2".to_string())
            ]
        );
        assert_eq!(
            section1.get("key2").unwrap().as_ref().unwrap(),
            &vec![Value::String("value3".to_string())]
        );
        assert!(section1.get("key3").unwrap().is_none()); // Missing key should be None

        // Check section2
        let section2 = &result["section2"];
        assert_eq!(section2.len(), 1);
        assert_eq!(
            section2.get("key4").unwrap().as_ref().unwrap(),
            &vec![
                Value::String("value4".to_string()),
                Value::String("value5".to_string())
            ]
        );

        // Check section3 (empty section from template)
        let section3 = &result["section3"];
        assert_eq!(section3.len(), 0);
    }

    #[test]
    fn test_parse_document_as_strings() {
        // Document with mixed value types
        let input = "section1 key1: value1, 123 key2: 45.67, true\nsection2 key3: false, 789";
        let result = parse_document_as_strings(input, None);

        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_complex_document() {
        let input = "section1\n key1: value1, value2 key2: value3\nsection2\n key4: value4, value5";
        let result = parse_document_as_strings(input, None).unwrap();
        println!("{:?}", result);
    }

    #[test]
    fn test_parse_this_sections_basic() {
        let input = "section1 key1: value1, value2 key2: value3\nsection2 key3: value4, value5";
        let titles = vec!["section1".to_string()];
        let result = parse_this_sections(input, titles).unwrap();

        assert_eq!(result.len(), 1);
        assert!(result.contains_key("section1"));
        assert!(!result.contains_key("section2"));

        let section1 = &result["section1"];
        assert_eq!(section1.len(), 2);
        assert!(section1.get("key1").unwrap().is_some());
        assert!(section1.get("key2").unwrap().is_some());
    }
}

#[cfg(test)]
mod tests2 {
    use crate::Utils::task_parser::{
        DocumentParser, parse_document_as_strings, parse_this_sections, parse_title,
    };
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{Read, Write};
    use tempfile::tempdir;
    #[test]
    fn test_parse_document_as_strings_basic() {
        let input =
            "title1\n key1: value1, value2\n key2: value3 \ntitle2\n key3: value4, value5\n";
        let result = parse_document_as_strings(input, None).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("title1"));
        assert!(result.contains_key("title2"));

        let title1_map = &result["title1"];
        assert_eq!(
            title1_map.get("key1").unwrap().as_ref().unwrap(),
            &vec!["value1".to_string(), "value2".to_string()]
        );
        assert_eq!(
            title1_map.get("key2").unwrap().as_ref().unwrap(),
            &vec!["value3".to_string()]
        );

        let title2_map = &result["title2"];
        assert_eq!(
            title2_map.get("key3").unwrap().as_ref().unwrap(),
            &vec!["value4".to_string(), "value5".to_string()]
        );
    }

    #[test]
    fn test_parse_document_as_strings_mixed_types() {
        let input = "title1\n key1: 1.0, 2.5\n key2: true, false \ntitle2\n key3: hello, world\n";
        let result = parse_document_as_strings(input, None).unwrap();

        assert_eq!(result.len(), 2);

        let title1_map = &result["title1"];
        assert_eq!(
            title1_map.get("key1").unwrap().as_ref().unwrap(),
            &vec!["1".to_string(), "2.5".to_string()]
        ); // TODO: Should be float?
        assert_eq!(
            title1_map.get("key2").unwrap().as_ref().unwrap(),
            &vec!["true".to_string(), "false".to_string()]
        );

        let title2_map = &result["title2"];
        assert_eq!(
            title2_map.get("key3").unwrap().as_ref().unwrap(),
            &vec!["hello".to_string(), "world".to_string()]
        );
    }

    #[test]
    fn test_parse_document_as_strings_with_template() {
        let input = "title1\n key1: 1, 2\n";

        let mut template = HashMap::new();
        let mut title1_map = HashMap::new();
        title1_map.insert("key1".to_string(), None);
        title1_map.insert("key2".to_string(), None);
        template.insert("title1".to_string(), title1_map);
        template.insert("title2".to_string(), HashMap::new());

        let result = parse_document_as_strings(input, Some(template)).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("title1"));
        assert!(result.contains_key("title2"));

        let title1_map = &result["title1"];
        assert_eq!(
            title1_map.get("key1").unwrap().as_ref().unwrap(),
            &vec!["1".to_string(), "2".to_string()]
        );
        assert!(title1_map.get("key2").unwrap().is_none());
    }

    #[test]
    fn test_parse_document_as_strings_empty() {
        let input = "";
        let res = parse_document_as_strings(input, None).is_err();
        assert_eq!(res, true);
    }

    #[test]
    fn test_parse_document_as_strings_malformed() {
        let input = "title1\n key1: value1, \n invalid structure";
        let result = parse_document_as_strings(input, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_document_from_file() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_doc.txt");

        // Create a test document file
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "config").unwrap();
        writeln!(file, "  port: 8080").unwrap();
        writeln!(file, "  hosts: localhost, 127.0.0.1").unwrap();
        writeln!(file, "logging").unwrap();
        writeln!(file, "  level: debug, info, warn").unwrap();
        writeln!(file, "  enabled: true").unwrap();

        // Read the file
        let mut file = File::open(&file_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        // Parse the document
        let result = parse_document_as_strings(&contents, None).unwrap();

        // Verify the parsed content
        assert_eq!(result.len(), 2);

        let config = &result["config"];
        assert_eq!(
            config.get("port").unwrap().as_ref().unwrap(),
            &vec!["8080".to_string()]
        );
        assert_eq!(
            config.get("hosts").unwrap().as_ref().unwrap(),
            &vec!["localhost".to_string(), "127.0.0.1".to_string()]
        );

        let logging = &result["logging"];
        assert_eq!(
            logging.get("level").unwrap().as_ref().unwrap(),
            &vec!["debug".to_string(), "info".to_string(), "warn".to_string()]
        );
        assert_eq!(
            logging.get("enabled").unwrap().as_ref().unwrap(),
            &vec!["true".to_string()]
        );
    }
    #[test]
    fn test_from_r_string() {
        let input = r#"system_config
          memory: 8GB, 16GB, 32GB
          cpu_cores: 4, 8, 16
          ssd_capacity: 256GB, 512GB, 1TB
        "#;
        let title = parse_title(input);
        assert!(title.is_ok());
        // assert_eq!(title, "system_config");
    }
    #[test]
    fn test_parse_document_complex() {
        let input = r#"
         system_config
          memory: 8GB, 16GB, 32GB
          cpu_cores: 4, 8, 16
          ssd_capacity: 256GB, 512GB, 1TB
        
        network_settings
          ip_addresses: 192.168.1.1, 10.0.0.1
          ports: 80, 443, 8080
          protocols: http, https, tcp
        
        user_preferences
          theme: dark, light
          font_size: 12, 14, 16
          notifications: true, false
        "#;

        let result = parse_document_as_strings(input, None).unwrap();

        assert_eq!(result.len(), 3);

        // Check system_config section
        let system = &result["system_config"];
        assert_eq!(
            system.get("memory").unwrap().as_ref().unwrap(),
            &vec!["8GB".to_string(), "16GB".to_string(), "32GB".to_string()]
        );
        assert_eq!(
            system.get("cpu_cores").unwrap().as_ref().unwrap(),
            &vec!["4".to_string(), "8".to_string(), "16".to_string()]
        );
        assert_eq!(
            system.get("ssd_capacity").unwrap().as_ref().unwrap(),
            &vec!["256GB".to_string(), "512GB".to_string(), "1TB".to_string()]
        );

        // Check network_settings section
        let network = &result["network_settings"];
        assert_eq!(
            network.get("ip_addresses").unwrap().as_ref().unwrap(),
            &vec!["192.168.1.1".to_string(), "10.0.0.1".to_string()]
        );
        assert_eq!(
            network.get("ports").unwrap().as_ref().unwrap(),
            &vec!["80".to_string(), "443".to_string(), "8080".to_string()]
        );
        assert_eq!(
            network.get("protocols").unwrap().as_ref().unwrap(),
            &vec!["http".to_string(), "https".to_string(), "tcp".to_string()]
        );

        // Check user_preferences section
        let prefs = &result["user_preferences"];
        assert_eq!(
            prefs.get("theme").unwrap().as_ref().unwrap(),
            &vec!["dark".to_string(), "light".to_string()]
        );
        assert_eq!(
            prefs.get("font_size").unwrap().as_ref().unwrap(),
            &vec!["12".to_string(), "14".to_string(), "16".to_string()]
        );
        assert_eq!(
            prefs.get("notifications").unwrap().as_ref().unwrap(),
            &vec!["true".to_string(), "false".to_string()]
        );
    }

    #[test]
    fn test_parse_document_with_special_characters() {
        let input = r#"
        database
          connection_string: mongodb://user:pass@localhost:27017, postgresql://postgres:password@localhost:5432
          timeout: 30s, 1m, 5m
        
        paths
          log_dir: /var/log/app, C:\logs\app
          data_dir: /var/data, D:\data
        "#;

        let result = parse_document_as_strings(input, None).unwrap();

        assert_eq!(result.len(), 2);

        let db = &result["database"];
        assert_eq!(
            db.get("connection_string").unwrap().as_ref().unwrap(),
            &vec![
                "mongodb://user:pass@localhost:27017".to_string(),
                "postgresql://postgres:password@localhost:5432".to_string()
            ]
        );

        let paths = &result["paths"];
        assert_eq!(
            paths.get("log_dir").unwrap().as_ref().unwrap(),
            &vec!["/var/log/app".to_string(), "C:\\logs\\app".to_string()]
        );
    }

    #[test]
    fn test_parse_document_with_comments() {
        let input = r#"
        database
          connection_string: mongodb://user:pass@localhost:27017, postgresql://postgres:password@localhost:5432
          timeout: 30s, 1m, 5m
        // this is comment
        # this is comment
        paths
          log_dir: /var/log/app, C:\logs\app
          data_dir: /var/data, D:\data
        "#;

        let result = parse_document_as_strings(input, None).unwrap();

        assert_eq!(result.len(), 2);

        let db = &result["database"];
        assert_eq!(
            db.get("connection_string").unwrap().as_ref().unwrap(),
            &vec![
                "mongodb://user:pass@localhost:27017".to_string(),
                "postgresql://postgres:password@localhost:5432".to_string()
            ]
        );

        let paths = &result["paths"];
        assert_eq!(
            paths.get("log_dir").unwrap().as_ref().unwrap(),
            &vec!["/var/log/app".to_string(), "C:\\logs\\app".to_string()]
        );
    }

    #[test]
    fn test_parse_this_sections_with_comments() {
        let input = r#"
        // Comment line
        config
          host: localhost
          port: 8080
        # Another comment
        database
          host: db.example.com
        % Yet another comment
        logging
          level: debug
        "#;

        let titles = vec!["config".to_string(), "logging".to_string()];
        let result = parse_this_sections(input, titles).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("logging"));
        assert!(!result.contains_key("database"));
    }

    #[test]
    fn test_document_parser_basic() {
        let input = "config host: localhost port: 8080\nlogging level: debug";
        let mut parser = DocumentParser::new(input.to_string());

        let result = parser.parse_document().unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("logging"));
        assert!(parser.is_success());
    }

    #[test]
    fn test_document_parser_with_template() {
        let input = "config host: localhost";
        let mut template = HashMap::new();
        let mut config_section = HashMap::new();
        config_section.insert("host".to_string(), None);
        config_section.insert("port".to_string(), None);
        template.insert("config".to_string(), config_section);

        let mut parser = DocumentParser::new(input.to_string()).with_template(template);
        let result = parser.parse_document_as().unwrap();

        let config = &result["config"];
        assert_eq!(config.len(), 2);
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_none());
    }

    #[test]
    fn test_document_parser_as_strings() {
        let input = "config port: 8080 debug: true";
        let mut parser = DocumentParser::new(input.to_string());

        let result = parser.parse_document_as_strings().unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("config"));

        let config = &result["config"];
        assert_eq!(
            config.get("port").unwrap().as_ref().unwrap(),
            &vec!["8080".to_string()]
        );
        assert_eq!(
            config.get("debug").unwrap().as_ref().unwrap(),
            &vec!["true".to_string()]
        );
    }

    #[test]
    fn test_document_parser_this_sections() {
        let input = "config host: localhost\ndatabase port: 5432\nlogging level: debug";
        let mut parser = DocumentParser::new(input.to_string());

        let titles = vec!["config".to_string(), "logging".to_string()];
        let result = parser.parse_this_sections(titles).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("logging"));
        assert!(!result.contains_key("database"));
    }

    #[test]
    fn test_document_parser_error_handling() {
        let input = "invalid input";
        let mut parser = DocumentParser::new(input.to_string());

        let result = parser.parse_document();
        assert!(result.is_err());
        assert!(!parser.is_success());
        assert!(parser.get_error().is_some());
    }

    #[test]
    fn test_document_parser_reset() {
        let input = "config host: localhost";
        let mut parser = DocumentParser::new(input.to_string());

        parser.parse_document().unwrap();
        assert!(parser.is_success());

        parser.reset();
        assert!(!parser.is_success());
        assert!(parser.get_result().is_none());
        assert!(parser.get_error().is_none());
    }

    #[test]
    fn test_document_parser_set_input() {
        let mut parser = DocumentParser::new("invalid".to_string());
        parser.parse_document().unwrap_err();
        assert!(!parser.is_success());

        parser.set_input("config host: localhost".to_string());
        parser.parse_document().unwrap();
        assert!(parser.is_success());
    }
}
#[cfg(test)]
mod template_tests {

    use crate::Utils::task_parser::{Value, parse_document_as, parse_this_sections};
    use std::collections::HashMap;
    #[test]
    fn test_parse_document_as_with_complex_template() {
        let mut template = HashMap::new();

        let mut section1 = HashMap::new();
        section1.insert("existing_key".to_string(), None);
        section1.insert("missing_key1".to_string(), None);
        section1.insert("missing_key2".to_string(), None);
        template.insert("config".to_string(), section1);

        let mut section2 = HashMap::new();
        section2.insert("db_host".to_string(), None);
        section2.insert("db_port".to_string(), None);
        section2.insert("db_name".to_string(), None);
        template.insert("database".to_string(), section2);

        let mut section3 = HashMap::new();
        section3.insert("level".to_string(), None);
        section3.insert("format".to_string(), None);
        section3.insert("output".to_string(), None);
        template.insert("logging".to_string(), section3);

        let input = "config existing_key: value1, value2\nlogging level: debug, info";
        let result = parse_document_as(input, Some(template)).unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("database"));
        assert!(result.contains_key("logging"));

        let config = &result["config"];
        assert_eq!(config.len(), 3);
        assert!(config.get("existing_key").unwrap().is_some());
        assert!(config.get("missing_key1").unwrap().is_none());
        assert!(config.get("missing_key2").unwrap().is_none());

        let database = &result["database"];
        assert_eq!(database.len(), 3);
        assert!(database.get("db_host").unwrap().is_none());
        assert!(database.get("db_port").unwrap().is_none());
        assert!(database.get("db_name").unwrap().is_none());

        let logging = &result["logging"];
        assert_eq!(logging.len(), 3);
        assert!(logging.get("level").unwrap().is_some());
        assert!(logging.get("format").unwrap().is_none());
        assert!(logging.get("output").unwrap().is_none());
    }

    #[test]
    fn test_parse_document_as_nested_template_structure() {
        let mut template = HashMap::new();

        let mut server_config = HashMap::new();
        server_config.insert("host".to_string(), None);
        server_config.insert("port".to_string(), None);
        server_config.insert("ssl_enabled".to_string(), None);
        server_config.insert("ssl_cert".to_string(), None);
        server_config.insert("ssl_key".to_string(), None);
        server_config.insert("max_connections".to_string(), None);
        server_config.insert("timeout".to_string(), None);
        template.insert("server".to_string(), server_config);

        let mut auth_config = HashMap::new();
        auth_config.insert("enabled".to_string(), None);
        auth_config.insert("providers".to_string(), None);
        auth_config.insert("oauth_client_id".to_string(), None);
        auth_config.insert("oauth_secret".to_string(), None);
        auth_config.insert("ldap_server".to_string(), None);
        auth_config.insert("ldap_base_dn".to_string(), None);
        template.insert("authentication".to_string(), auth_config);

        let mut monitoring = HashMap::new();
        monitoring.insert("metrics_enabled".to_string(), None);
        monitoring.insert("health_check_interval".to_string(), None);
        monitoring.insert("alert_endpoints".to_string(), None);
        template.insert("monitoring".to_string(), monitoring);

        let input = r#"
        server
          host: localhost, 0.0.0.0
          port: 8080, 8443
          ssl_enabled: true
        
        authentication
          enabled: true
          providers: oauth, ldap
        "#;

        let result = parse_document_as(input, Some(template)).unwrap();

        assert_eq!(result.len(), 3);

        let server = &result["server"];
        assert_eq!(server.len(), 7);
        assert!(server.get("host").unwrap().is_some());
        assert!(server.get("port").unwrap().is_some());
        assert!(server.get("ssl_enabled").unwrap().is_some());
        assert!(server.get("ssl_cert").unwrap().is_none());
        assert!(server.get("ssl_key").unwrap().is_none());
        assert!(server.get("max_connections").unwrap().is_none());
        assert!(server.get("timeout").unwrap().is_none());

        let auth = &result["authentication"];
        assert_eq!(auth.len(), 6);
        assert!(auth.get("enabled").unwrap().is_some());
        assert!(auth.get("providers").unwrap().is_some());
        assert!(auth.get("oauth_client_id").unwrap().is_none());

        let monitoring = &result["monitoring"];
        assert_eq!(monitoring.len(), 3);
        assert!(monitoring.get("metrics_enabled").unwrap().is_none());
        assert!(monitoring.get("health_check_interval").unwrap().is_none());
        assert!(monitoring.get("alert_endpoints").unwrap().is_none());
    }

    #[test]
    fn test_parse_document_as_empty_template_sections() {
        let mut template = HashMap::new();
        template.insert("empty_section1".to_string(), HashMap::new());
        template.insert("empty_section2".to_string(), HashMap::new());

        let mut populated_section = HashMap::new();
        populated_section.insert("key1".to_string(), None);
        template.insert("populated".to_string(), populated_section);

        let input = "populated key1: value1";
        let result = parse_document_as(input, Some(template)).unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result["empty_section1"].len(), 0);
        assert_eq!(result["empty_section2"].len(), 0);
        assert_eq!(result["populated"].len(), 1);
        assert!(result["populated"]["key1"].is_some());
    }

    #[test]
    fn test_parse_document_as_with_mixed_value_types_template() {
        let mut template = HashMap::new();

        let mut config = HashMap::new();
        config.insert("numbers".to_string(), None);
        config.insert("booleans".to_string(), None);
        config.insert("strings".to_string(), None);
        config.insert("missing_mixed".to_string(), None);
        template.insert("mixed_types".to_string(), config);

        let input = "mixed_types numbers: 1, 2.5, 42 booleans: true, false strings: hello, world";
        let result = parse_document_as(input, Some(template)).unwrap();

        let mixed = &result["mixed_types"];
        assert_eq!(mixed.len(), 4);

        let numbers = mixed.get("numbers").unwrap().as_ref().unwrap();
        assert_eq!(numbers.len(), 3);
        assert_eq!(numbers[0], Value::Integer(1));
        assert_eq!(numbers[1], Value::Float(2.5));
        assert_eq!(numbers[2], Value::Integer(42));

        let booleans = mixed.get("booleans").unwrap().as_ref().unwrap();
        assert_eq!(booleans[0], Value::Boolean(true));
        assert_eq!(booleans[1], Value::Boolean(false));

        assert!(mixed.get("missing_mixed").unwrap().is_none());
    }

    #[test]
    fn test_parse_document_as_large_template() {
        let mut template = HashMap::new();

        let mut app = HashMap::new();
        app.insert("name".to_string(), None);
        app.insert("version".to_string(), None);
        app.insert("debug".to_string(), None);
        app.insert("environment".to_string(), None);
        template.insert("application".to_string(), app);

        let mut primary_db = HashMap::new();
        primary_db.insert("host".to_string(), None);
        primary_db.insert("port".to_string(), None);
        primary_db.insert("database".to_string(), None);
        primary_db.insert("username".to_string(), None);
        primary_db.insert("password".to_string(), None);
        primary_db.insert("pool_size".to_string(), None);
        template.insert("primary_database".to_string(), primary_db);

        let mut cache_db = HashMap::new();
        cache_db.insert("redis_host".to_string(), None);
        cache_db.insert("redis_port".to_string(), None);
        cache_db.insert("redis_db".to_string(), None);
        cache_db.insert("ttl".to_string(), None);
        template.insert("cache".to_string(), cache_db);

        let mut security = HashMap::new();
        security.insert("jwt_secret".to_string(), None);
        security.insert("jwt_expiry".to_string(), None);
        security.insert("cors_origins".to_string(), None);
        security.insert("rate_limit".to_string(), None);
        template.insert("security".to_string(), security);

        let mut external = HashMap::new();
        external.insert("email_service".to_string(), None);
        external.insert("sms_service".to_string(), None);
        external.insert("payment_gateway".to_string(), None);
        external.insert("analytics_key".to_string(), None);
        template.insert("external_services".to_string(), external);

        let input = r#"
        application
          name: MyApp
          debug: true
        
        primary_database
          host: localhost
          port: 5432
        
        security
          cors_origins: http://localhost:3000, https://myapp.com
        "#;

        let result = parse_document_as(input, Some(template)).unwrap();

        assert_eq!(result.len(), 5);
        assert!(result.contains_key("application"));
        assert!(result.contains_key("primary_database"));
        assert!(result.contains_key("cache"));
        assert!(result.contains_key("security"));
        assert!(result.contains_key("external_services"));

        let app = &result["application"];
        assert_eq!(app.len(), 4);
        assert!(app.get("name").unwrap().is_some());
        assert!(app.get("debug").unwrap().is_some());
        assert!(app.get("version").unwrap().is_none());
        assert!(app.get("environment").unwrap().is_none());

        let cache = &result["cache"];
        assert_eq!(cache.len(), 4);
        assert!(cache.values().all(|v| v.is_none()));

        let external = &result["external_services"];
        assert_eq!(external.len(), 4);
        assert!(external.values().all(|v| v.is_none()));
    }

    #[test]
    fn test_parse_document_as_no_template() {
        let input = "section1 key1: value1, value2 key2: value3\nsection2 key3: value4, value5";
        let result = parse_document_as(input, None).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("section1"));
        assert!(result.contains_key("section2"));

        let section1 = &result["section1"];
        assert_eq!(section1.len(), 2);
        assert!(section1.get("key1").unwrap().is_some());
        assert!(section1.get("key2").unwrap().is_some());
    }

    #[test]
    fn test_parse_this_sections() {
        let input = r#"
        config
          host: localhost
          port: 8080
        
        database
          host: db.example.com
          port: 5432
        
        logging
          level: debug
          output: file
        "#;

        // Parse only config and logging sections
        let titles = vec!["config".to_string(), "logging".to_string()];
        let result = parse_this_sections(input, titles).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("logging"));
        assert!(!result.contains_key("database")); // Should not be included

        let config = &result["config"];
        assert_eq!(config.len(), 2);
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_some());

        let logging = &result["logging"];
        assert_eq!(logging.len(), 2);
        assert!(logging.get("level").unwrap().is_some());
        assert!(logging.get("output").unwrap().is_some());
    }

    #[test]
    fn test_parse_this_sections_missing_titles() {
        let input = "config host: localhost port: 8080";

        // Request sections that don't exist
        let titles = vec!["missing1".to_string(), "missing2".to_string()];
        let result = parse_this_sections(input, titles).unwrap();

        assert_eq!(result.len(), 0); // Should be empty
    }

    #[test]
    fn test_parse_this_sections_partial_match() {
        let input = r#"
        section1
          key1: value1
        
        section2
          key2: value2
        
        section3
          key3: value3
        "#;

        // Request mix of existing and non-existing sections
        let titles = vec![
            "section1".to_string(),
            "missing".to_string(),
            "section3".to_string(),
        ];
        let result = parse_this_sections(input, titles).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("section1"));
        assert!(result.contains_key("section3"));
        assert!(!result.contains_key("section2"));
        assert!(!result.contains_key("missing"));
    }
}

#[cfg(test)]
mod pseudonym_tests {
    use crate::Utils::task_parser::{DocumentParser, Value, invert_vec_map, parse_document};
    use std::collections::HashMap;

    #[test]
    fn test_invert_vec_map() {
        let mut input = HashMap::new();
        input.insert(
            "real_name1".to_string(),
            vec!["alias1".to_string(), "alias2".to_string()],
        );
        input.insert("real_name2".to_string(), vec!["alias3".to_string()]);

        let result = invert_vec_map(&input);

        assert_eq!(result.len(), 3);
        assert_eq!(result.get("alias1").unwrap(), "real_name1");
        assert_eq!(result.get("alias2").unwrap(), "real_name1");
        assert_eq!(result.get("alias3").unwrap(), "real_name2");
    }

    #[test]
    fn test_header_pseudonyms() {
        let input = "cfg host: localhost port: 8080\ndb_config host: db.example.com";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up header pseudonyms
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert(
            "config".to_string(),
            vec!["cfg".to_string(), "configuration".to_string()],
        );
        header_pseudonyms.insert(
            "database".to_string(),
            vec!["db_config".to_string(), "db".to_string()],
        );

        parser.with_pseudonims(Some(header_pseudonyms), None);
        let result = parser.parse_document().unwrap();

        // Should have real names, not aliases
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("database"));
        assert!(!result.contains_key("cfg"));
        assert!(!result.contains_key("db_config"));

        let config = &result["config"];
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_some());
    }

    #[test]
    fn test_field_name_pseudonyms() {
        let input = "config hostname: localhost prt: 8080";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up field name pseudonyms
        let mut field_pseudonyms = HashMap::new();
        field_pseudonyms.insert(
            "host".to_string(),
            vec!["hostname".to_string(), "server".to_string()],
        );
        field_pseudonyms.insert("port".to_string(), vec!["prt".to_string(), "p".to_string()]);

        parser.with_pseudonims(None, Some(field_pseudonyms));
        let result = parser.parse_document().unwrap();

        let config = &result["config"];
        assert_eq!(config.len(), 2);
        // Should have real names, not aliases
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_some());
        assert!(!config.contains_key("hostname"));
        assert!(!config.contains_key("prt"));
    }

    #[test]
    fn test_both_header_and_field_pseudonyms() {
        let input = "cfg hostname: localhost prt: 8080\ndb_config srv: db.example.com";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up both types of pseudonyms
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert("config".to_string(), vec!["cfg".to_string()]);
        header_pseudonyms.insert("database".to_string(), vec!["db_config".to_string()]);

        let mut field_pseudonyms = HashMap::new();
        field_pseudonyms.insert("host".to_string(), vec!["hostname".to_string()]);
        field_pseudonyms.insert("port".to_string(), vec!["prt".to_string()]);
        field_pseudonyms.insert("server".to_string(), vec!["srv".to_string()]);

        parser.with_pseudonims(Some(header_pseudonyms), Some(field_pseudonyms));
        let result = parser.parse_document().unwrap();

        // Check headers are converted
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("database"));

        // Check field names are converted
        let config = &result["config"];
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_some());

        let database = &result["database"];
        assert!(database.get("server").unwrap().is_some());
    }

    #[test]
    fn test_pseudonyms_with_template() {
        let input = "cfg hostname: localhost";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up template with real names
        let mut template = HashMap::new();
        let mut config_section = HashMap::new();
        config_section.insert("host".to_string(), None);
        config_section.insert("port".to_string(), None);
        template.insert("config".to_string(), config_section);

        parser = parser.with_template(template);

        // Set up pseudonyms
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert("config".to_string(), vec!["cfg".to_string()]);

        let mut field_pseudonyms = HashMap::new();
        field_pseudonyms.insert("host".to_string(), vec!["hostname".to_string()]);

        parser.with_pseudonims(Some(header_pseudonyms), Some(field_pseudonyms));
        let result = parser.parse_document_as().unwrap();
        print!("{:?}", result);
        // Should have real names and template structure
        let config = &result["config"];
        assert_eq!(config.len(), 2);
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_none()); // From template
    }

    #[test]
    fn test_pseudonyms_no_match() {
        let input = "config host: localhost port: 8080";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up pseudonyms that don't match anything
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert("database".to_string(), vec!["db".to_string()]);

        let mut field_pseudonyms = HashMap::new();
        field_pseudonyms.insert("server".to_string(), vec!["srv".to_string()]);

        parser.with_pseudonims(Some(header_pseudonyms), Some(field_pseudonyms));
        let result = parser.parse_document().unwrap();

        // Should keep original names when no pseudonyms match
        assert!(result.contains_key("config"));
        let config = &result["config"];
        assert!(config.get("host").unwrap().is_some());
        assert!(config.get("port").unwrap().is_some());
    }

    #[test]
    fn test_pseudonyms_with_parse_this_sections() {
        let input = "cfg hostname: localhost\ndb_config srv: db.example.com\nlogging level: debug";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up pseudonyms
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert("config".to_string(), vec!["cfg".to_string()]);
        header_pseudonyms.insert("database".to_string(), vec!["db_config".to_string()]);

        let mut field_pseudonyms = HashMap::new();
        field_pseudonyms.insert("host".to_string(), vec!["hostname".to_string()]);
        field_pseudonyms.insert("server".to_string(), vec!["srv".to_string()]);

        parser.with_pseudonims(Some(header_pseudonyms), Some(field_pseudonyms));

        // Parse only specific sections (using real names)
        let titles = vec!["config".to_string(), "database".to_string()];
        let result = parser.parse_this_sections(titles).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("config"));
        assert!(result.contains_key("database"));
        assert!(!result.contains_key("logging")); // Should not be included

        let config = &result["config"];
        assert!(config.get("host").unwrap().is_some());

        let database = &result["database"];
        assert!(database.get("server").unwrap().is_some());
    }

    #[test]
    fn test_find_actual_pseudonyms_conversion() {
        let input = "cfg host: localhost\ndb_config port: 5432\nlogging level: debug";
        let mut parser = DocumentParser::new(input.to_string());

        // Set up header pseudonyms
        let mut header_pseudonyms = HashMap::new();
        header_pseudonyms.insert(
            "config".to_string(),
            vec!["cfg".to_string(), "configuration".to_string()],
        );
        header_pseudonyms.insert(
            "database".to_string(),
            vec!["db_config".to_string(), "db".to_string()],
        );

        parser.with_pseudonims(Some(header_pseudonyms), None);

        // Parse the full document first to discover actual pseudonyms
        let full_doc = parse_document(&input).unwrap();

        // Test finding actual pseudonyms used in the document
        let real_names = vec![
            "config".to_string(),
            "database".to_string(),
            "unknown".to_string(),
        ];
        let actual_pseudonyms = parser.find_actual_pseudonyms(&full_doc, real_names);

        assert_eq!(actual_pseudonyms.len(), 3);
        assert_eq!(actual_pseudonyms[0], "cfg"); // This pseudonym is actually in the document
        assert_eq!(actual_pseudonyms[1], "db_config"); // This pseudonym is actually in the document
        assert_eq!(actual_pseudonyms[2], "unknown"); // No mapping and not in document, should stay the same
    }
}

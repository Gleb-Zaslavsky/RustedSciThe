/// parse document with structure like " title1 key1: value1, value2 key2: value2 title2 key3:value3, value4" which has titles and
/// pairs key-vector of values.  user defines hashmap HashMap<String HashMap<String, Option<Vec<Type>> >> to parse data into.
/// If some fielf i.e field_x not found in the resultiong map it will be field_x:None.
///
///
/*
 code to parse document with structure like " title1 key1: value1, value2 key2: value2 title2 key3:value3, value4" which has titles and
 pairs key-vector of values. 1) user defines hashmap HashMap<String HashMap<String, Option<Vec<Type>> >> to parse data into.
 If some fielf i.e field_x not found in the resultiong map it will be field_x:None 2) Function must parse this data
  into HashMap<String HashMap<String, Option<Vec<Type>> >>

*/
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{alpha1, alphanumeric1, multispace0, space0, line_ending},
    combinator::{map, map_res, recognize, not},
    multi::{many0, many1, separated_list0},
    sequence::{delimited, pair, separated_pair, terminated},
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;

type DocumentMap = HashMap<String, SectionMap>;
type SectionMap = HashMap<String, Option<Vec<Value>>>;
/// enum to represent different value types:
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Float(f64),
    Integer(i64),
    Boolean(bool),
    // Add other types as needed
}
#[allow(dead_code)]
impl Value {
    // Helper functions to access different value types
    pub fn as_string(&self) -> Option<&String> {
        if let Value::String(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        if let Value::Float(f) = self {
            Some(*f)
        } else {
            None
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        if let Value::Integer(i) = self {
            Some(*i)
        } else {
            None
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        if let Value::Boolean(b) = self {
            Some(*b)
        } else {
            None
        }
    }
    // Try to convert to string representation
    pub fn to_string_value(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Float(f) => f.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Boolean(b) => b.to_string(),
        }
    }
}

// Implement Display for Value
impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::String(s) => write!(f, "{}", s),
            Value::Float(val) => write!(f, "{}", val),
            Value::Integer(val) => write!(f, "{}", val),
            Value::Boolean(val) => write!(f, "{}", val),
        }
    }
}

/// Parses a title (word characters without spaces)
fn parse_title(input: &str) -> IResult<&str, String> {
    let parser = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ));

    let mut parser = map(parser, String::from);

    // Ignore trailing whitespace, newline characters, and semicolons

    let (input, result) = parser.parse(input)?;

    let input = input.trim();
    //  println!("Title: {}", input);
    Ok((input, result))
}

/// Parses a key (word characters without spaces)
fn parse_key(input: &str) -> IResult<&str, String> {
    let parser = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ));

    let mut parser = map(parser, String::from);
    let (input, result) = parser.parse(input)?;

    Ok((input, result))
}

fn parse_value(input: &str) -> IResult<&str, Value> {
    // Parse a single value - excluding commas, whitespace, newlines, and semicolons
    let value_parser = take_while1(|c: char| !matches!(c, ',' | ' ' | '\t' | '\n' | ';'));
    let mut value_parser = map_res(value_parser, |s: &str| -> Result<Value, String> {
        let s = s.trim();
        // Try parsing as different types in order
        if let Ok(val) = s.parse::<i64>() {
            Ok(Value::Integer(val))
        } else if let Ok(val) = s.parse::<f64>() {
            Ok(Value::Float(val))
        } else if let Ok(val) = s.parse::<bool>() {
            Ok(Value::Boolean(val))
        } else {
            Ok(Value::String(s.to_string()))
        }
    });

    let (input, result) = value_parser.parse(input)?;

    Ok((input, result))
}

fn parse_value_list(input: &str) -> IResult<&str, Vec<Value>> {
    let (input, _) = multispace0(input)?;
    // Parse the comma-separated values
    // delimited function - Matches an object from the first parser and discards it, then gets an object from the second parser,
    // and finally matches an object from the third parser and discards it. So here we deop spaces from the beginning and end of
    // the line, and then parse the key and values.
    let separator_coma = delimited(space0, tag(","), space0);
    let mut value_parser = separated_list0(separator_coma, parse_value);
    let (input, result) = value_parser.parse(input)?;

    Ok((input, result))
}

/// Parses a key-value pair where value is a list
fn parse_key_value_pair(input: &str) -> IResult<&str, (String, Vec<Value>)> {
    // Parse the colon with optional whitespace
    // space0 - Matches zero or more whitespace characters.
    let colon_separator = delimited(space0, tag(":"), space0);
    // separated_pair - Matches two objects from the first and second parsers, respectively, and discards them.
    let mut parser = separated_pair(parse_key, colon_separator, parse_value_list);
    let (input, result) = parser.parse(input)?;
    Ok((input.trim(), result))
}

/// Parses a section with a title and multiple key-value pairs
fn parse_section(input: &str) -> IResult<&str, (String, HashMap<String, Vec<Value>>)> {
    let (input, _) = space0(input)?;
    let (input, title) = parse_title(input)?;

    // Modified to handle both spaces and newlines after the title
    let (input, _) = multispace0(input)?;
    let mut parser = many1(terminated(parse_key_value_pair, space0));
    let (input, pairs) = parser.parse(input)?;

    let mut section_map = HashMap::new();
    for (key, values) in pairs {
        section_map.insert(key, values);
    }

    Ok((input, (title, section_map)))
}

/// Filters out comment lines (starting with //, #, %, or ;)
fn filter_comments(input: &str) -> String {
    input
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with("//") && !trimmed.starts_with('#') && 
            !trimmed.starts_with('%') && !trimmed.starts_with(';') &&
            !trimmed.is_empty()
        })
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Parses the entire document into a HashMap
pub fn parse_document(input: &str) -> IResult<&str, DocumentMap> {
  //  let filtered_input = filter_comments(input);
   // let (input, _) = multispace0(&filtered_input)?;
    // Use many1 instead of separated_list0 to parse sections
    // and ensure each section is properly terminated
    let mut parser = many1(delimited(
        space0, // Allow optional whitespace before a section
        parse_section,
        multispace0, // Allow optional whitespace after a section
    ));

    let (input, sections) = parser.parse(input)?;

    let mut result = HashMap::new();
    for (title, section_map) in sections.into_iter() {
        let mut title_map = HashMap::new();
        for (key, values) in section_map {
            title_map.insert(key, Some(values));
        }
        result.insert(title, title_map);
    }

    Ok((input, result))
}

/// Parses a document and merges with a template HashMap, ensuring all expected keys exist
fn parse_document_with_template(
    input: &str,
    template: &HashMap<String, HashMap<String, Option<Vec<Value>>>>,
) -> Result<DocumentMap, String> {
    let parse_result = parse_document(input);

    match parse_result {
        Ok((remaining, mut parsed)) => {
            if !remaining.trim().is_empty() {
                return Err(format!(
                    "Failed to parse entire document. Remaining: '{}'",
                    remaining
                ));
            }

            // Ensure all expected titles and keys exist
            for (title, keys_map) in template {
                if !parsed.contains_key(title) {
                    parsed.insert(title.clone(), HashMap::new());
                }

                let section_map = parsed.get_mut(title).unwrap();
                for key in keys_map.keys() {
                    if !section_map.contains_key(key) {
                        section_map.insert(key.clone(), None);
                    }
                }
            }

            Ok(parsed)
        }
        Err(e) => Err(format!("Parsing error: {:?}", e)),
    }
}

/// Helper function to parse a document
pub fn parse_document_as(
    input: &str,
    template: Option<DocumentMap>,
) -> Result<DocumentMap, String> {
    match template {
        Some(template) => parse_document_with_template(input, &template),
        None => match parse_document(input) {
            Ok((remaining, parsed)) => {
                if !remaining.trim().is_empty() {
                    return Err(format!(
                        "Failed to parse entire document. Remaining: '{}'",
                        remaining
                    ));
                }
                Ok(parsed)
            }
            Err(e) => Err(format!("Parsing error: {:?}", e)),
        },
    }
}

/// function to parse document into HashMap<String, HashMap<String, Option<Vec<String>>>
/// the differ is that nested hashmap has type not  HashMap<String, Option<Vec<Value>>>
/// but HashMap<String, Option<Vec<String>>
/// Parses the document and converts all values to strings
pub fn parse_document_as_strings(
    input: &str,
    template: Option<HashMap<String, HashMap<String, Option<Vec<String>>>>>,
) -> Result<HashMap<String, HashMap<String, Option<Vec<String>>>>, String> {
    // First parse with our Value enum
    let value_result = parse_document_as(input, None);

    match value_result {
        Ok(value_map) => {
            // Convert the Value map to a String map
            let mut string_map = HashMap::new();

            for (title, section_map) in value_map {
                let mut string_section = HashMap::new();

                for (key, value_opt) in section_map {
                    let string_values = value_opt.map(|values| {
                        values
                            .into_iter()
                            .map(|v| v.to_string_value())
                            .collect::<Vec<String>>()
                    });

                    string_section.insert(key, string_values);
                }

                string_map.insert(title, string_section);
            }

            // If a template was provided, ensure all expected keys exist
            if let Some(template) = template {
                for (title, keys_map) in template {
                    if !string_map.contains_key(&title) {
                        string_map.insert(title.clone(), HashMap::new());
                    }

                    let section_map = string_map.get_mut(&title).unwrap();
                    for key in keys_map.keys() {
                        if !section_map.contains_key(key) {
                            section_map.insert(key.clone(), None);
                        }
                    }
                }
            }

            Ok(string_map)
        }
        Err(e) => Err(e),
    }
}

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
    use super::*;

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
        let (remaining, doc) = parse_document(input).unwrap();
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

        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_document_with_mixed_types() {
        // Document with mixed value types
        let input = "section1 key1: value1, 123 key2: 45.67, true\nsection2 key3: false, 789";
        let (remaining, doc) = parse_document(input).unwrap();

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

        assert_eq!(remaining, "");
    }

    #[test]
    fn test_parse_document_with_multiple_line_breaks() {
        // Document with multiple line breaks between sections
        let input = "section1 key1: value1, value2\n\n\nsection2 key2: value3, value4";
        let (remaining, doc) = parse_document(input).unwrap();

        assert_eq!(doc.len(), 2);
        assert!(doc.contains_key("section1"));
        assert!(doc.contains_key("section2"));
        assert_eq!(remaining, "");
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
}

#[cfg(test)]
mod tests2 {
    use super::*;
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
}

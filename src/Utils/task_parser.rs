//! # Task Parser Module
//!
//! A comprehensive document parser for structured text with sections and key-value pairs.
//! Parses documents with format: `title1 key1: value1, value2 key2: value2 title2 key3: value3, value4`
//! into nested HashMaps with type-safe value handling and advanced features.
//!
//! ## Core Functionality
//! - **Structured Parsing**: Converts text documents into `HashMap<String, HashMap<String, Option<Vec<Value>>>>`
//! - **Type Safety**: Supports multiple value types (String, Float, Integer, Boolean, Vector, Optional)
//! - **Template System**: Pre-defines expected structure with `TemplateMap` for validation
//! - **Pseudonym Support**: Allows aliases for section headers and field names
//! - **Selective Parsing**: Parse only specific sections by title
//! - **Comment Filtering**: Automatically filters lines starting with //, #, %, or ;
//!
//! ## Main Structures and Enums
//!
//! ### `DocumentParser`
//! Main parser struct with state management and error handling:
//! - `new(input)` - Create parser with input text
//! - `with_template(template)` - Set parsing template
//! - `with_pseudonims(headers, fields)` - Configure aliases
//! - `parse_document()` - Parse entire document
//! - `parse_document_as()` - Parse with template support
//! - `parse_this_sections(titles)` - Parse specific sections only
//! - `parse_document_as_strings()` - Parse all values as strings
//!
//! ### `Value` Enum
//! Type-safe value representation with variants:
//! - `String(String)` - Text values
//! - `Float(f64)` - Floating point numbers
//! - `Integer(i64)` - Signed integers
//! - `Usize(usize)` - Unsigned size type
//! - `Vector(Vec<f64>)` - Arrays of floats in `[1.0, 2.0, 3.0]` format
//! - `Boolean(bool)` - True/false values
//! - `Optional(Option<Box<Value>>)` - Nullable values with `None` or `Some(value)` syntax
//!
//! Helper methods: `as_string()`, `as_float()`, `as_integer()`, `as_boolean()`, `as_vector()`, etc.
//!
//! ### `TemplateType` Enum
//! Simplified enum for template specification without data:
//! - Used in `TemplateMap` to define expected structure
//! - Variants match `Value` enum but without actual data
//!
//! ## Key Functions
//!
//! ### Core Parsing Functions
//! - `parse_document(input)` - Parse entire document into DocumentMap
//! - `parse_document_as(input, template)` - Parse with optional template
//! - `parse_this_sections(input, titles)` - Parse only specified sections
//! - `parse_document_as_strings(input, template)` - Parse all values as strings
//!
//! ### Low-level Parsers (using nom crate)
//! - `parse_title(input)` - Parse section headers
//! - `parse_key(input)` - Parse field names
//! - `parse_value(input)` - Parse single values with type detection
//! - `parse_value_list(input)` - Parse comma-separated value lists
//! - `parse_key_value_pair(input)` - Parse `key: value1, value2` pairs
//! - `parse_section(input)` - Parse complete sections
//!
//! ### Utility Functions
//! - `invert_vec_map(map)` - Convert `HashMap<String, Vec<String>>` to `HashMap<String, String>`
//! - `filter_comments(input)` - Remove comment lines from input
//!
//! ## Type Aliases
//! - `DocumentMap` = `HashMap<String, SectionMap>`
//! - `SectionMap` = `HashMap<String, Option<Vec<Value>>>`
//! - `TemplateMap` = `HashMap<String, HashMap<String, Option<Vec<TemplateType>>>>`
//!
//! ## Non-obvious Features and Tips
//!
//! ### 1. Bracket-Aware Value Parsing
//! The `parse_single_value()` function tracks bracket `[]` and parenthesis `()` depth to correctly
//! parse complex nested structures like vectors and optional values without breaking on internal commas.
//!
//! ### 2. Pseudonym System Complexity
//! - One real name can map to multiple pseudonyms
//! - Parser must discover which pseudonyms are actually used in input
//! - `find_actual_pseudonyms()` method resolves real names to actual pseudonyms in document
//! - Templates are converted from real names to pseudonyms before parsing
//!
//! ### 3. Template-Pseudonym Integration
//! When both templates and pseudonyms are used:
//! - Templates are defined with real names
//! - Input uses pseudonyms
//! - `convert_template_to_pseudonyms()` dynamically adapts template structure
//! - Results are converted back to real names after parsing
//!
//! ### 4. Optional Value Parsing
//! Supports both `None` and `Some(value)` syntax with recursive inner value parsing.
//! The parser can handle nested optionals like `Some(Some(42))`.
//!
//! ### 5. Vector Parsing
//! Vectors use `[value1, value2, value3]` syntax and are parsed as `Vec<f64>`.
//! Empty vectors `[]` are supported.
//!
//! ### 6. Comment Filtering
//! Automatically removes lines starting with `//`, `#`, `%`, or `;` before parsing.
//! Empty lines are also filtered out.
//!
//! ### 7. Error Handling Strategy
//! - Parser stores last error in `error` field
//! - `is_success()` method checks both error state and result presence
//! - Graceful fallbacks when pseudonym resolution fails
//!
//! ### 8. Memory Efficiency
//! - Uses `Box<Value>` for optional values to reduce enum size
//! - Reuses parsing functions through composition
//! - Minimal string allocations during parsing
//!
//! ## Usage Examples
//!
//! ```rust, ignore
//! // Basic parsing
//! let mut parser = DocumentParser::new("section1 key1: value1, value2".to_string());
//! let result = parser.parse_document().unwrap();
//!
//! // With template
//! let template = HashMap::from([("section1".to_string(),
//!     HashMap::from([("key1".to_string(), Some(vec![TemplateType::String]))]));
//! let mut parser = DocumentParser::new(input).with_template(template);
//! let result = parser.parse_document_as().unwrap();
//!
//! // With pseudonyms
//! let mut parser = DocumentParser::new(input);
//! parser.with_pseudonims(
//!     Some(HashMap::from([("section1".to_string(), vec!["sec1".to_string()])])),
//!     Some(HashMap::from([("key1".to_string(), vec!["k1".to_string()])]))
//! );
//! let result = parser.parse_document().unwrap();
//! ```
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, multispace0, space0},
    combinator::{map, recognize},
    error::Error,
    multi::{many0, many1, separated_list0},
    sequence::{delimited, pair, separated_pair, terminated},
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum TemplateType {
    String,
    Float,
    Integer,
    Usize,
    Vector,
    Boolean,
    Optional,
}

pub type DocumentMap = HashMap<String, SectionMap>;
type SectionMap = HashMap<String, Option<Vec<Value>>>;
type TemplateMap = HashMap<String, HashMap<String, Option<Vec<TemplateType>>>>;

/// Document parser struct that wraps parsing functionality
#[derive(Debug, Clone)]
pub struct DocumentParser {
    pub input: String,
    pub result: Option<DocumentMap>,
    pub template: Option<TemplateMap>,
    pub headers_pseudonims: Option<HashMap<String, String>>,
    pub field_name_pseudonims: Option<HashMap<String, String>>,
    pub string_result: Option<HashMap<String, HashMap<String, Option<Vec<String>>>>>,
    pub error: Option<String>,
}

impl DocumentParser {
    /// Create a new DocumentParser with input
    pub fn new(input: String) -> Self {
        Self {
            input,
            result: None,
            template: None,
            headers_pseudonims: None,
            field_name_pseudonims: None,
            string_result: None,
            error: None,
        }
    }

    /// Set template for parsing
    pub fn with_template(mut self, template: TemplateMap) -> Self {
        self.template = Some(template);
        self
    }

    /// Parse the document
    pub fn parse_document(&mut self) -> Result<&DocumentMap, String> {
        match parse_document(&self.input) {
            Ok(result) => {
                let result = self.to_real_names(Some(result)).unwrap();
                self.result = Some(result);
                self.error = None;
                Ok(self.result.as_ref().unwrap())
            }
            Err(e) => {
                self.error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Parse document with template support
    pub fn parse_document_as(&mut self) -> Result<&DocumentMap, String> {
        // Convert template to use actual pseudonyms if pseudonyms are configured
        let template_for_parsing = if let Some(template) = &self.template {
            Some(self.convert_template_to_pseudonyms(template))
        } else {
            None
        };

        match parse_document_as(&self.input, template_for_parsing) {
            Ok(result) => {
                let result = self.to_real_names(Some(result)).unwrap();
                self.result = Some(result);
                self.error = None;
                Ok(self.result.as_ref().unwrap())
            }
            Err(e) => {
                self.error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Parse document as strings
    pub fn parse_document_as_strings(
        &mut self,
    ) -> Result<&HashMap<String, HashMap<String, Option<Vec<String>>>>, String> {
        let template = self.template.as_ref().map(|t| {
            t.iter()
                .map(|(k, v)| {
                    let string_map = v.iter().map(|(key, _)| (key.clone(), None)).collect();
                    (k.clone(), string_map)
                })
                .collect()
        });

        match parse_document_as_strings(&self.input, template) {
            Ok(result) => {
                self.string_result = Some(result);
                self.error = None;
                Ok(self.string_result.as_ref().unwrap())
            }
            Err(e) => {
                self.error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Parse specific sections by titles (titles should be real names)
    pub fn parse_this_sections(&mut self, titles: Vec<String>) -> Result<&DocumentMap, String> {
        // First parse the entire document to discover which pseudonyms are actually used
        let full_doc = match parse_document(&self.input) {
            Ok(doc) => doc,
            Err(e) => return Err(e),
        };

        // Find actual pseudonyms used in input that correspond to requested real names
        let parsing_titles = self.find_actual_pseudonyms(&full_doc, titles);

        match parse_this_sections(&self.input, parsing_titles) {
            Ok(result) => {
                let result = self.to_real_names(Some(result)).unwrap();
                self.result = Some(result);
                self.error = None;
                Ok(self.result.as_ref().unwrap())
            }
            Err(e) => {
                self.error = Some(e.clone());
                Err(e)
            }
        }
    }

    /// Get the parsed result
    pub fn get_result(&self) -> Option<&DocumentMap> {
        self.result.as_ref()
    }

    /// Get the string result
    pub fn get_string_result(
        &self,
    ) -> Option<&HashMap<String, HashMap<String, Option<Vec<String>>>>> {
        self.string_result.as_ref()
    }

    /// Get the last error
    pub fn get_error(&self) -> Option<&String> {
        self.error.as_ref()
    }

    /// Check if parsing was successful
    pub fn is_success(&self) -> bool {
        self.error.is_none() && (self.result.is_some() || self.string_result.is_some())
    }

    /// Reset the parser state
    pub fn reset(&mut self) {
        self.result = None;
        self.string_result = None;
        self.error = None;
    }

    /// Update input and reset state
    pub fn set_input(&mut self, input: String) {
        self.input = input;
        self.reset();
    }

    /// Load settings from file
    pub fn setting_from_file(&mut self, path: Option<PathBuf>) -> Result<(), String> {
        let file_path = match path {
            Some(p) => p,
            None => {
                // Search for file starting with "problem" in current directory
                let current_dir = std::env::current_dir()
                    .map_err(|e| format!("Failed to get current directory: {}", e))?;

                let entries = fs::read_dir(&current_dir)
                    .map_err(|e| format!("Failed to read directory: {}", e))?;

                let mut problem_file = None;
                for entry in entries {
                    let entry =
                        entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
                    let file_name = entry.file_name();
                    let file_name_str = file_name.to_string_lossy();

                    if file_name_str.starts_with("problem") && file_name_str.ends_with(".txt") {
                        problem_file = Some(entry.path());
                        break;
                    }
                }

                problem_file.ok_or_else(|| "No file starting with 'problem' and ending with '.txt' found in current directory".to_string())?
            }
        };

        let content = fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read file {:?}: {}", file_path, e))?;

        self.set_input(content);
        Ok(())
    }

    /// pseudonims are given as "real name of field":{vec!["names user can use for the name of field"]}
    /// Converts HashMap<String, Vec<String>> to HashMap<String, String>
    /// Each element of the Vec becomes a key, and the original key becomes its value
    pub fn with_pseudonims(
        &mut self,
        headers_pseudonims: Option<HashMap<String, Vec<String>>>,
        field_name_pseudonims: Option<HashMap<String, Vec<String>>>,
    ) {
        if let Some(headers_pseudonims) = headers_pseudonims {
            let headers_pseudonims = invert_vec_map(&headers_pseudonims);
            self.headers_pseudonims = Some(headers_pseudonims);
        }

        if let Some(field_name_pseudonims) = field_name_pseudonims {
            let field_name_pseudonims = invert_vec_map(&field_name_pseudonims);
            self.field_name_pseudonims = Some(field_name_pseudonims);
        }
    }

    pub fn to_real_names(&self, result: Option<DocumentMap>) -> Option<DocumentMap> {
        if let Some(mut doc_map) = result {
            // Handle header pseudonyms
            if let Some(headers_pseudonims) = &self.headers_pseudonims {
                let mut new_doc_map = HashMap::new();
                for (header, section_map) in doc_map {
                    let real_header = headers_pseudonims
                        .get(&header)
                        .map(|s| s.clone())
                        .unwrap_or(header);
                    new_doc_map.insert(real_header, section_map);
                }
                doc_map = new_doc_map;
            }

            // Handle field name pseudonyms
            if let Some(field_name_pseudonims) = &self.field_name_pseudonims {
                for (_, section_map) in doc_map.iter_mut() {
                    let mut new_section_map = HashMap::new();
                    for (field_name, values) in section_map.drain() {
                        let real_field_name = field_name_pseudonims
                            .get(&field_name)
                            .map(|s| s.clone())
                            .unwrap_or(field_name);
                        new_section_map.insert(real_field_name, values);
                    }
                    *section_map = new_section_map;
                }
            }

            Some(doc_map)
        } else {
            None
        }
    }
    /// Find actual pseudonyms used in the document that correspond to requested real names
    pub fn find_actual_pseudonyms(
        &self,
        full_doc: &DocumentMap,
        real_names: Vec<String>,
    ) -> Vec<String> {
        if let Some(headers_pseudonims) = &self.headers_pseudonims {
            let mut result = Vec::new();

            for real_name in real_names {
                // Find which pseudonym (if any) is actually used in the document for this real name
                let mut found_pseudonym = None;

                // Check all pseudonyms that map to this real name
                for (pseudonym, mapped_real_name) in headers_pseudonims {
                    if mapped_real_name == &real_name && full_doc.contains_key(pseudonym) {
                        found_pseudonym = Some(pseudonym.clone());
                        break;
                    }
                }

                // Use the found pseudonym or the real name if no pseudonym is found
                result.push(found_pseudonym.unwrap_or(real_name));
            }

            result
        } else {
            real_names
        }
    }

    // Convert template from real names to pseudonyms for parsing
    fn convert_template_to_pseudonyms(&self, template: &TemplateMap) -> TemplateMap {
        if let (Some(headers_pseudonims), Some(field_pseudonims)) =
            (&self.headers_pseudonims, &self.field_name_pseudonims)
        {
            // First parse input to discover actual pseudonyms
            let full_doc = match parse_document(&self.input) {
                Ok(doc) => doc,
                Err(_) => return template.clone(), // If parsing fails, return original template
            };

            let mut converted_template = HashMap::new();

            for (real_header, real_fields) in template {
                // Find actual pseudonym for this header
                let actual_header_pseudonym = headers_pseudonims
                    .iter()
                    .find(|&(_, &ref real)| real == real_header)
                    .and_then(|(pseudo, _)| {
                        if full_doc.contains_key(pseudo) {
                            Some(pseudo.clone())
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| real_header.clone());

                // Convert field names to pseudonyms
                let mut converted_fields = HashMap::new();
                for (real_field, template_type) in real_fields {
                    let actual_field_pseudonym = field_pseudonims
                        .iter()
                        .find(|&(_, &ref real)| real == real_field)
                        .map(|(pseudo, _)| pseudo.clone())
                        .unwrap_or_else(|| real_field.clone());

                    converted_fields.insert(actual_field_pseudonym, template_type.clone());
                }

                converted_template.insert(actual_header_pseudonym, converted_fields);
            }

            converted_template
        } else {
            template.clone()
        }
    }
    pub fn input_to_lower_case(&mut self) {
        let mut result = String::new();
        let mut chars = self.input.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch.is_alphabetic() {
                let mut word = String::new();
                word.push(ch);

                // Collect the rest of the word
                while let Some(&next_ch) = chars.peek() {
                    if next_ch.is_alphabetic() {
                        word.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }

                // Preserve "Some" and "None", lowercase everything else
                if word == "Some" || word == "None" {
                    result.push_str(&word);
                } else {
                    result.push_str(&word.to_lowercase());
                }
            } else {
                result.push(ch);
            }
        }

        self.input = result;
    }

    pub fn keys_to_lower_case(&mut self, exception: Option<Vec<String>>) {
        if let Some(ref result) = self.result {
            let mut new_result = DocumentMap::new();
            for (key, value) in result {
                let new_key = key.to_lowercase();
                //  if outer key is in exception vector than don't do anything with nested key -
                // just copy to the result map
                let mut new_section_map = SectionMap::new();
                for (nested_key, nested_value) in value {
                    let new_nested_key = if let Some(ref exceptions) = exception {
                        if exceptions.contains(key) {
                            nested_key.clone() // Don't change nested key if outer key is in exceptions
                        } else {
                            nested_key.to_lowercase()
                        }
                    } else {
                        nested_key.to_lowercase()
                    };
                    new_section_map.insert(new_nested_key, nested_value.clone());
                }
                new_result.insert(new_key, new_section_map);
            }
            self.result = Some(new_result);
        }
    }
}
/// Converts HashMap<String, Vec<String>> to HashMap<String, String>
/// Each element of the Vec becomes a key, and the original key becomes its value
pub fn invert_vec_map(map: &HashMap<String, Vec<String>>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    for (key, vec) in map {
        for val in vec {
            result.insert(val.clone(), key.clone());
        }
    }
    result
}
/// enum to represent different value types:
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Float(f64),
    Integer(i64),
    Usize(usize),
    Vector(Vec<f64>),
    Boolean(bool),
    Optional(Option<Box<Value>>),
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
    pub fn as_usize(&self) -> Option<usize> {
        if let Value::Integer(i) = self {
            Some(*i as usize)
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

    pub fn as_vector(&self) -> Option<&Vec<f64>> {
        if let Value::Vector(v) = self {
            Some(v)
        } else {
            None
        }
    }

    pub fn as_optional(&self) -> Option<&Option<Box<Value>>> {
        if let Value::Optional(opt) = self {
            Some(opt)
        } else {
            None
        }
    }
    pub fn as_option_string(&self) -> Option<&String> {
        if let Value::Optional(opt) = self {
            if let Some(inner) = opt {
                if let Value::String(s) = inner.as_ref() {
                    return Some(s);
                }
            }
        }
        None
    }

    pub fn as_option_float(&self) -> Option<f64> {
        if let Value::Optional(opt) = self {
            if let Some(inner) = opt {
                if let Value::Float(f) = inner.as_ref() {
                    return Some(*f);
                }
            }
        }
        None
    }
    pub fn as_option_integer(&self) -> Option<i64> {
        if let Value::Optional(opt) = self {
            if let Some(inner) = opt {
                if let Value::Integer(i) = inner.as_ref() {
                    return Some(*i);
                }
            }
        }
        None
    }

    pub fn as_option_usize(&self) -> Option<usize> {
        if let Value::Optional(opt) = self {
            if let Some(inner) = opt {
                if let Value::Integer(i) = inner.as_ref() {
                    return Some(*i as usize);
                }
            }
        }
        None
    }
    // Try to convert to string representation
    pub fn to_string_value(&self) -> String {
        match self {
            Value::String(s) => s.clone(),
            Value::Float(f) => f.to_string(),
            Value::Integer(i) => i.to_string(),
            Value::Usize(i) => i.to_string(),
            Value::Vector(v) => v
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<String>>()
                .join(", "),
            Value::Boolean(b) => b.to_string(),
            Value::Optional(opt) => {
                if let Some(inner) = opt {
                    inner.to_string_value()
                } else {
                    "None".to_string()
                }
            }
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
            Value::Usize(val) => write!(f, "{}", val),
            Value::Vector(val) => write!(f, "{:?}", val),
            Value::Boolean(val) => write!(f, "{}", val),
            Value::Optional(opt) => {
                if let Some(inner) = opt {
                    write!(f, "{}", inner)
                } else {
                    write!(f, "None")
                }
            }
        }
    }
}

/// Parses a title (word characters without spaces)
pub fn parse_title(input: &str) -> IResult<&str, String> {
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
pub fn parse_key(input: &str) -> IResult<&str, String> {
    let parser = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ));

    let mut parser = map(parser, String::from);
    let (input, result) = parser.parse(input)?;

    Ok((input, result))
}

fn parse_single_value(input: &str) -> IResult<&str, &str> {
    let mut chars = input.char_indices();
    let mut bracket_depth = 0;
    let mut paren_depth = 0;
    let mut end_pos = 0;

    for (pos, ch) in chars {
        match ch {
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            ',' | ' ' | '\t' | '\n' | ';' if bracket_depth == 0 && paren_depth == 0 => {
                break;
            }
            _ => {}
        }
        end_pos = pos + ch.len_utf8();
    }

    if end_pos == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::TakeWhile1,
        )));
    }

    Ok((&input[end_pos..], &input[..end_pos]))
}

pub fn parse_value(input: &str) -> IResult<&str, Value> {
    let (remaining, value_str) = parse_single_value(input)?;
    let s = value_str.trim();

    let value = if s == "None" {
        Value::Optional(None)
    } else if s.starts_with("Some(") && s.ends_with(')') {
        // Extract inner value from Some(...)
        let inner = &s[5..s.len() - 1];
        // Recursively parse the inner value
        if let Ok(val) = inner.parse::<i64>() {
            Value::Optional(Some(Box::new(Value::Integer(val))))
        } else if let Ok(val) = inner.parse::<f64>() {
            Value::Optional(Some(Box::new(Value::Float(val))))
        } else if let Ok(val) = inner.parse::<bool>() {
            Value::Optional(Some(Box::new(Value::Boolean(val))))
        } else if let Ok(val) = inner.parse::<usize>() {
            Value::Optional(Some(Box::new(Value::Usize(val))))
        } else {
            Value::Optional(Some(Box::new(Value::String(inner.to_string()))))
        }
    } else if s.starts_with('[') && s.ends_with(']') {
        // Parse as vector of floats
        let inner = &s[1..s.len() - 1];
        if inner.is_empty() {
            Value::Vector(vec![])
        } else {
            let values = inner
                .split(',')
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            Value::Vector(values)
        }
    } else if let Ok(val) = s.parse::<i64>() {
        Value::Integer(val)
    } else if let Ok(val) = s.parse::<f64>() {
        Value::Float(val)
    } else if let Ok(val) = s.parse::<bool>() {
        Value::Boolean(val)
    } else if let Ok(val) = s.parse::<usize>() {
        Value::Usize(val)
    } else {
        Value::String(s.to_string())
    };

    Ok((remaining, value))
}

pub fn parse_value_list(input: &str) -> IResult<&str, Vec<Value>> {
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
pub fn parse_key_value_pair(input: &str) -> IResult<&str, (String, Vec<Value>)> {
    // Parse the colon with optional whitespace
    // space0 - Matches zero or more whitespace characters.
    let colon_separator = delimited(space0, tag(":"), space0);
    // separated_pair - Matches two objects from the first and second parsers, respectively, and discards them.
    let mut parser = separated_pair(parse_key, colon_separator, parse_value_list);
    let (input, result) = parser.parse(input)?;
    Ok((input.trim(), result))
}

/// Parses a section with a title and multiple key-value pairs
pub fn parse_section(input: &str) -> IResult<&str, (String, HashMap<String, Vec<Value>>)> {
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
            !trimmed.starts_with("//")
                && !trimmed.starts_with('#')
                && !trimmed.starts_with('%')
                && !trimmed.starts_with(';')
                && !trimmed.is_empty()
        })
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Parses the entire document into a HashMap
pub fn parse_document(input: &str) -> Result<DocumentMap, String> {
    let filtered_input = filter_comments(input);
    let (remaining_input, _) = multispace0::<&str, Error<&str>>(filtered_input.as_str())
        .map_err(|e| format!("Parsing error: {:?}", e))?;

    // Use many1 instead of separated_list0 to parse sections
    // and ensure each section is properly terminated
    let mut parser = many1(delimited(
        space0, // Allow optional whitespace before a section
        parse_section,
        multispace0, // Allow optional whitespace after a section
    ));

    let (remaining, sections) = parser
        .parse(remaining_input)
        .map_err(|e| format!("Parsing error: {:?}", e))?;

    if !remaining.trim().is_empty() {
        return Err(format!(
            "Failed to parse entire document. Remaining: '{}'",
            remaining
        ));
    }

    let mut result = HashMap::new();
    for (title, section_map) in sections.into_iter() {
        let mut title_map = HashMap::new();
        for (key, values) in section_map {
            title_map.insert(key, Some(values));
        }
        result.insert(title, title_map);
    }

    Ok(result)
}

/// Parses a document and merges with a template HashMap, ensuring all expected keys exist
pub fn parse_document_with_template(
    input: &str,
    template: &TemplateMap,
) -> Result<DocumentMap, String> {
    let mut parsed = parse_document(input)?;

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

/// Helper function to parse a document
pub fn parse_document_as(
    input: &str,
    template: Option<TemplateMap>,
) -> Result<DocumentMap, String> {
    match template {
        Some(template) => parse_document_with_template(input, &template),
        None => parse_document(input),
    }
}

/// Parses only specific sections by their titles
pub fn parse_this_sections(input: &str, titles: Vec<String>) -> Result<DocumentMap, String> {
    let filtered_input = filter_comments(input);
    let (mut remaining_input, _) = multispace0::<&str, Error<&str>>(filtered_input.as_str())
        .map_err(|e| format!("Parsing error: {:?}", e))?;

    let mut result = HashMap::new();

    // Keep parsing sections until input is exhausted
    while !remaining_input.trim().is_empty() {
        // Try to parse a section
        match parse_section(remaining_input) {
            Ok((remaining, (title, section_map))) => {
                // Only add to result if title is in the requested titles
                if titles.contains(&title) {
                    let mut title_map = HashMap::new();
                    for (key, values) in section_map {
                        title_map.insert(key, Some(values));
                    }
                    result.insert(title, title_map);
                }

                // Update remaining input
                let (new_remaining, _) = multispace0::<&str, Error<&str>>(remaining)
                    .map_err(|e| format!("Parsing error: {:?}", e))?;
                remaining_input = new_remaining;
            }
            Err(_) => {
                // If we can't parse a section, we're done
                break;
            }
        }
    }

    Ok(result)
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

#[cfg(test)]

mod tests {
    use rand::rand_core::le;
    use toml::value;

    use super::*;
    #[test]
    fn close_to_life_example1() {
        let input = "
        solver_settings
        scheme: forward
        method: Dense
        strategy: Damped
        linear_sys_method: None
        abs_tolerance: 1e-6
        max_iterations: 100
        loglevel: Some(info)
        bounds
        z: -10.0, 10.0
        y: -7.0, 7.0
        ";
        let res = parse_document(input);
        let max_iterations = res
            .clone()
            .unwrap()
            .get("solver_settings")
            .unwrap()
            .get("max_iterations")
            .unwrap()
            .clone();
        println!("max_iterations: {:?}", max_iterations);
        assert!(max_iterations.is_some());
        let max_iter_value = max_iterations.unwrap()[0].clone();
        println!("max_iter_value: {:?}", max_iter_value);
        let max_iter = max_iter_value.as_usize().unwrap();
        assert!(max_iter == 100);
        let bounds = res.clone().unwrap().get("bounds").unwrap().clone();
        let y_bounds = bounds.get("y").unwrap().clone().unwrap();
        let y0 = y_bounds[0].clone().as_float().unwrap();
        let y1 = y_bounds[1].clone().as_float().unwrap();
        assert_eq!(y0, -7.0);
        assert_eq!(y1, 7.0);
        println!("bounds: {:?}", bounds);
        // assert!(max_iter == 100);
        println!("res {:?}", res);
        let bounds_from_map: HashMap<String, (f64, f64)> = bounds
            .iter()
            .map(|(key, value)| {
                let binding = value.clone().unwrap();
                let value0 = binding[0].as_float().unwrap();
                let value1 = binding[1].as_float().unwrap();
                (key.to_owned(), (value0, value1))
            })
            .collect();
        println!("bounds_from_map: {:?}", bounds_from_map);
        assert!(res.is_ok());
    }
}

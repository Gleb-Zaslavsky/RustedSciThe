use criterion::{ criterion_group, criterion_main, Criterion, black_box};
use RustedSciThe::Examples::bvp_examples::bvp_examples; 

/*
fn benchmark_bvp_examples(c: &mut Criterion) {
    let mut group = c.benchmark_group("BVP Examples");

    for i in 0..6 {
        group.bench_function(format!("Example {}", i), |b| {
            b.iter(|| bvp_examples(black_box(i)))
        });
    }

    group.finish();
}

 */

 #[allow(dead_code)]
 fn bench_example_1(c: &mut Criterion) {
    c.bench_function("example_1", |b| b.iter(|| bvp_examples(1)));
}
#[allow(dead_code)]
fn bench_example_7(c: &mut Criterion) {
    let mut group = c.benchmark_group("My benchmark group");
    group.sample_size(400);

    group.bench_function("BVP example 7", |b| b.iter(|| black_box(bvp_examples(7)) ));
    group.finish();
}// 321 239
#[allow(dead_code)]
fn bench_example_7v1(c: &mut Criterion) {
    let mut group = c.benchmark_group("My benchmark group");
    group.sample_size(400);
   
    group.bench_function("BVP example 7v1", |b| b.iter(|| bvp_examples(7)));
    group.finish();
} // 230 230
#[allow(dead_code)]
fn bench_example_6(c: &mut Criterion) {
    let mut group = c.benchmark_group("My benchmark group");
    group.sample_size(400);
  
    group.bench_function("BVP example 6", |b| b.iter(|| black_box(bvp_examples(6)) ));
    group.finish();
}// 135
#[allow(dead_code)]
fn bench_example_6v1(c: &mut Criterion) {
    let mut group = c.benchmark_group("My benchmark group");
    group.sample_size(400);

    group.bench_function("BVP example 6v1", |b| b.iter(|| bvp_examples(6)));
    group.finish();
} //  264


criterion_group!(benches,  bench_example_7v1);
criterion_main!(benches);



// 4CPU
//   non_modified 
//              black_box       no black_box
//  example 7:      (241,239)                (231,241)
//  example 6:      (158, 154)               (153, 154)  
//    modified 
//  example 7:      (219, 220)                      (217, 218)
//  example 6:        (142, 141)                    (144, 146)
//20 CPU
//   non_modified 
//              black_box       no black_box
//  example 7:        (90, 92)             (91, 91)
//  example 6:       (60, 60)            (59,60 )   
//    modified
//              black_box       no black_box
//  example 7:      (88, 88 )              (88, 88)
//  example 6:      (59, 59)                (59, 60)   


/*
This code implements several key features to make the evaluation of symbolic functions fast and efficient:
1.
Expression Hashing:
The code uses a hashing mechanism (AHasher) to create unique identifiers for expressions. This allows for quick comparison and identification of identical subexpressions.
2.
Common Subexpression Elimination (CSE):
The remove_common_pairs function implements CSE, which identifies and eliminates redundant computations. This reduces the total number of operations needed during evaluation.
3.
Instruction-based Evaluation:
Instead of directly evaluating the expression tree, the code compiles the expression into a series of instructions (Instr enum). This allows for more efficient execution as it reduces the overhead of traversing the expression tree during each evaluation.
4.
Stack-based Computation:
The ExpressionEvaluator uses a stack to store intermediate results. This approach is cache-friendly and allows for efficient memory usage and access patterns.
5.
Vectorized Operations:
The instruction set includes vectorized operations for addition and multiplication (Instr::Add and Instr::Mul), which can operate on multiple operands at once, reducing the total number of instructions.
6.
Specialized Math Functions:
The code includes optimized implementations for common mathematical functions (exp, log, sin, cos, sqrt) through the BuiltinFun instruction.
7.
Compile-time Optimization:
The expression is analyzed and optimized when creating the ExpressionEvaluator, so the runtime evaluation can be as fast as possible.
8.
Reuse of Allocated Memory:
The evaluator reuses the same stack for multiple evaluations, reducing memory allocation overhead.
9.
Type Parameterization:
The code is generic over the numeric type T, allowing for specialized implementations for different numeric types (e.g., floating-point, arbitrary-precision) without changing the core algorithm.
10.
Efficient Data Structures:
The code uses efficient data structures like HashMap for quick lookups and Vec for contiguous memory storage of instructions and operands.
11.
Minimal Branching:
The instruction-based approach minimizes conditional branching during evaluation, which can be beneficial for CPU pipeline efficiency.
12.
Separation of Compilation and Evaluation:
The expression is compiled into an efficient form once, and then this compiled form can be evaluated multiple times with different inputs, amortizing the cost of optimization over multiple evaluations.
These features combine to create a system that can evaluate complex symbolic expressions quickly and efficiently, especially when the same expression needs to be evaluated multiple times with different inputs.


*/
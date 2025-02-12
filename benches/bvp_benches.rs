use criterion::{ criterion_group, criterion_main, Criterion};
use RustedSciThe::Examples::bvp_examples::bvp_examples; // Replace `your_crate_name` with the actual name of your crate
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


 fn bench_example_1(c: &mut Criterion) {
    c.bench_function("example_1", |b| b.iter(|| bvp_examples(1)));
}

fn bench_example_5(c: &mut Criterion) {
    c.bench_function("BVP example 5", |b| b.iter(|| bvp_examples(5)));
}

criterion_group!(benches,  bench_example_5);
criterion_main!(benches);

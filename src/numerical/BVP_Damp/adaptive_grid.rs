use crate::twopntbvproblem::TwoPntBVProblem;
use crate::twopntbvproblem::TwoPntBVPDomain;
use crate::twopntsolversetup::TwoPntSolverSetup;
//This Fortran subroutine refine is part of a larger program dealing with boundary value problems (BVPs) for differential equations. T
//he subroutine refines the grid used in solving the BVP by adding new points where necessary to improve the solution's accuracy.
/*
. Adaptive Grid Refinement
Error Estimation: After each iteration of the Newton method, estimate the local error in the solution. This can be done using techniques like:
Comparing solutions from different grid sizes.
Using a posteriori error estimates based on the residuals of the ODE.
Refinement Criteria: Define criteria for refining the grid based on the error estimates. Common strategies include:
Refining the grid where the error exceeds a certain threshold.
Coarsening the grid where the solution is smooth and the error is below a threshold.
. Grid Adjustment
Adding Points: If refinement is needed, new grid points are added in regions where the error is high. This can be done using:

Bisection: Splitting intervals into smaller sub-intervals.
Adaptive clustering: Placing more points in areas of high curvature or steep gradients.

Curvature: Areas where the solution exhibits high curvature or rapid changes (e.g., steep gradients) may require more grid points. This can be assessed by evaluating the second derivative of the solution.
Discontinuities: If the solution has discontinuities or sharp transitions, additional points should be added around these regions to capture the behavior accurately.

Let me explain the adaptive grid algorithm for solving boundary value problems (BVPs) using Newton’s method.
Key Steps of the Algorithm:
Initial Grid Setup:
Start with a coarse uniform grid
Let’s say initially N points: 
 represents local grid spacing
Error Estimation:
Solve BVP on current grid using Newton’s method
Compute local truncation error estimate  at each point
Common estimate: 
where  represents second derivatives
Grid Point Distribution:
The new grid points should be distributed according to the equidistribution principle:
Grid Adaptation:
Define tolerance  for maximum allowed error
For each interval :
If : Add new point(s)
If : Consider removing point(s)
Grid Smoothing:
Apply smoothing to avoid too rapid changes in grid spacing
Common ratio constraint: 
Solution Update:
Interpolate solution from old grid to new grid
Solve BVP on new grid using Newton’s method
Iteration:
Repeat steps 2-6 until:
Error estimates are below tolerance
Grid stops changing significantly
Maximum iteration count reached
Mathematical Framework:
The equidistribution principle can be expressed as:

where  is a weight function based on error estimates.
For mesh density function :

where  is a parameter controlling adaptation sensitivity.
Implementation Notes:
def adapt_grid(x, y, tol):
    # Compute error estimates
    tau = estimate_error(x, y)
    
    # Determine new grid points
    new_points = []
    for i in range(len(x)-1):
        if tau[i]*(x[i+1]-x[i]) > tol:
            # Add midpoint
            new_points.append((x[i] + x[i+1])/2)
    
    # Update grid
    x_new = sort(concatenate([x, new_points]))
    
    # Smooth grid if necessary
    x_new = smooth_grid(x_new)
    
    return x_new


Let me explain the equidistribution principle, which is a fundamental concept in adaptive mesh generation.
The Equidistribution Principle:
Basic Definition:
The principle states that the error should be equally distributed over all mesh elements. Mathematically:

where:
-  is a monitor/weight function
-  is a mesh interval
-  is the local integral value (same for all intervals)
Monitor Function Forms:
Common choices for  include:
a) Arc-length monitor function:
b) Curvature-based monitor:
c) Error-based monitor:

where  is local error estimate
Continuous Form:
The continuous equidistribution equation:

where  is computational coordinate
Discrete Implementation:
For a mesh with N points:
This leads to:

where:
-  is local mesh spacing
-  is discrete monitor function
-  is total integral of monitor function
Iterative Algorithm:
def equidistribute_mesh(x_old, w):
    N = len(x_old)
    x_new = np.zeros(N)
    x_new[0] = x_old[0]  # Fix left boundary
    
    # Compute cumulative integral of monitor function
    I = np.cumsum(w)
    I_total = I[-1]
    
    # Generate new mesh points
    for i in range(1, N-1):
        target = (i/(N-1)) * I_total
        # Find location where cumulative integral equals target
        x_new[i] = interpolate_to_value(I, x_old, target)
    
    x_new[-1] = x_old[-1]  # Fix right boundary
    return x_new
Properties and Advantages:
a) Mesh Density Relation:

*/
fn refine(
    this: &mut TwoPntBVProblem,
    error: &mut bool,
    setup: &TwoPntSolverSetup,
    text: i32,
    vars: &mut TwoPntBVPDomain,
    newx: &mut bool,
    success: &mut bool,
    mark: &mut [bool],
    ratio: &mut [f64],
    ratio1: &mut [f64],
    ratio2: &mut [f64],
    u: &mut [f64],
    buffer: &mut [f64],
    vary1: &mut [i32],
    vary2: &mut [i32],
    weight: &mut [i32],
) {
    let id = "REFINE:  ";
    let zero = 0.0;
    let one = 1.0;
    let half = 0.5;
    let mut word = String::new();
    let mut differ = 0.0;
    let mut left = 0.0;
    let mut length = 0.0;
    let mut lower = 0.0;
    let mut max_du = 0;
    let mut max_grad = 0;
    let mut maxmag = 0.0;
    let mut mean = 0.0;
    let mut more = 0;
    let mut most = 0;
    let mut new = 0;
    let mut old = 0;
    let mut padd = 0;
    let mut range = 0.0;
    let mut right = 0.0;
    let mut signif = 0;
    let mut temp = 0.0;
    let mut temp1 = 0.0;
    let mut temp2 = 0.0;
    let mut total = 0;
    let mut least = 0;
    let mut counted = 0;
    let mut act = 0;
    let mut itemp = 0;
    let mut j = 0;
    let mut k = 0;
    let mut leveld = 0;
    let mut levelm = 0;
    let mut former = 0;
    let mut toler0 = 0.0;
    let mut toler1 = 0.0;
    let mut toler2 = 0.0;

    leveld = setup.leveld - 1;
    levelm = setup.levelm - 1;
    padd = setup.ipadd;
    toler0 = setup.toler0;
    toler1 = setup.toler1;
    toler2 = setup.toler2;

    // Initialization: turn off all completion status flags.
    *error = false;
    *newx = false;
    *success = false;

    // Levelm printing.
    if levelm > 0 && text > 0 {
        println!("{}SELECT A GRID.", id);
    }

    // Check the arguments.
    vars.check_on_grid_update(error, id, text, padd);
    if *error {
        return;
    }

    // Check there is at least one variable that affects grid adaption
    counted = vars.active.iter().filter(|&&x| x).count();
    *error = counted < 1;
    if *error {
        if text > 0 {
            println!("{}ERROR.  THERE ARE NO ACTIVE COMPONENTS.", id);
        }
        return;
    }

    *error = toler0 < zero;
    if *error {
        if text > 0 {
            println!(
                "{}ERROR.  THE BOUNDS ON MAGNITUDE AND RELATIVE CHANGE OF MAGNITUDE FOR INSIGNIFICANT COMPONENTS MUST BE POSITIVE.\n\
                {:>10.2}  TOLER0, SIGNIFICANCE LEVEL",
                id, toler0
            );
        }
        return;
    }

    // Check tolerances in [0,1]
    *error = !(zero <= toler1 && toler1 <= one && zero <= toler2 && toler2 <= one);
    if *error {
        if text > 0 {
            println!(
                "{}ERROR.  THE BOUNDS ON RELATIVE CHANGES IN MAGNITUDE AND ANGLE MUST LIE BETWEEN 0 AND 1.\n\
                {:>10.2}  TOLER1\n\
                {:>10.2}  TOLER2",
                id, toler1, toler2
            );
        }
        return;
    }

    // Check monotonic
    counted = 0;
    for k in 0..vars.points - 1 {
        if vars.x[k] < vars.x[k + 1] {
            counted += 1;
        }
    }
    *error = counted != 0 && counted != vars.points - 1;
    if *error {
        if text > 0 {
            println!("{}ERROR.  THE GRID IS NOT ORDERED.", id);
        }
        return;
    }

    // at each interval, count the active, significant components that vary too greatly.
    act = 0; // number of active components
    signif = 0; // number of significant components: max(u)-min(u)>=tol*max(|u|)
    mark.iter_mut().for_each(|x| *x = false);
    ratio1.iter_mut().for_each(|x| *x = zero);
    ratio2.iter_mut().for_each(|x| *x = zero);
    vary1.iter_mut().for_each(|x| *x = 0);
    vary2.iter_mut().for_each(|x| *x = 0);

    // top of the loop over the components.
    for j in 0..vars.comps {
        if !vars.active[j] {
            continue;
        }
        act += 1;
        // find range and maximum magnitude of this component.
        lower = u[j * vars.points];
        upper = u[j * vars.points];
        for k in 1..vars.points {
            lower = lower.min(u[j * vars.points + k]);
            upper = upper.max(u[j * vars.points + k]);
        }
        range = upper - lower;
        maxmag = lower.abs().max(upper.abs());
        // decide whether the component is significant.
        if range.abs() <= toler0 * max(one, maxmag) {
            continue;
        }
        // this is a significant component.
        signif += 1;
        // at each interval, see whether the component'S CHANGE EXCEEDS SOME
        // fraction of the component'S GLOBAL CHANGE.
        for k in 0..vars.points - 1 {
            differ = (u[j * vars.points + k + 1] - u[j * vars.points + k]).abs();
            if range > zero {
                ratio1[k] = ratio1[k].max(differ / range);
            }
            if toler1 * range < differ {
                vary1[k] += 1;
            }
        }
        // find the global change of the component'S DERIVATIVE.
        temp = grad(u, &vars.x, j, 0);
        lower = temp;
        upper = temp;
        // at each interior point, see whether the derivative'S CHANGE
        // exceeds some fraction of the derivative'S GLOBAL CHANGE.
        for k in 1..vars.points - 1 {
            temp = grad(u, &vars.x, j, k);
            lower = lower.min(temp);
            upper = upper.max(temp);
        }
        range = upper - lower;
        right = grad(u, &vars.x, j, 0);
        for k in 1..vars.points - 1 {
            left = right;
            right = grad(u, &vars.x, j, k);
            differ = (left - right).abs();
            if range > zero {
                ratio2[k] = ratio2[k].max(differ / range);
            }
            if toler2 * range < differ {
                vary2[k] += 1;
            }
        }
    }

    // save the maximum ratios.
    ratio[0] = ratio1.iter().take(vars.points - 1).fold(0.0, |acc, &x| acc.max(x));
    ratio[1] = ratio2.iter().skip(1).take(vars.points - 2).fold(0.0, |acc, &x| acc.max(x));

    // ***** select the intervals to halve. *****
    // weight the intervals in which variations that are too large occur.
    most = 0;
    for k in 0..vars.points - 1 {
        weight[k] = vary1[k];
        if k > 0 {
            weight[k] += vary2[k];
        }
        if k < vars.points - 2 {
            weight[k] += vary2[k + 1];
        }
        if weight[k] > 0 {
            most += 1;
        }
    }

    // sort the weights using interchange sort.
    for k in 0..vars.points - 1 {
        for j in k + 1..vars.points - 1 {
            if weight[j] > weight[k] {
                itemp = weight[j];
                weight[j] = weight[k];
                weight[k] = itemp;
            }
        }
        if weight[k] == 0 {
            break;
        }
    }

    // find the least weight of intervals to halve.
    more = most.min(padd).min(vars.pmax - vars.points);
    if more > 0 {
        least = weight[more - 1];
    } else {
        least = 1 + weight[0];
    }

    // reconstruct the weights.
    for k in 0..vars.points - 1 {
        weight[k] = vary1[k];
        if k > 0 {
            weight[k] += vary2[k];
        }
        if k < vars.points - 2 {
            weight[k] += vary2[k + 1];
        }
    }

    // mark the intervals to halve.
    counted = 0;
    for k in 0..vars.points - 1 {
        if counted < more && least <= weight[k] {
            counted += 1;
            mark[k] = true;
        }
    }

    // ***** halve the intervals, if any. *****
    // total number of points in the new and old grid.
    total = vars.points + more;
    former = vars.points;

    if more > 0 {
        // check that the new points are not degenerate
        counted = 0;
        length = (vars.x[vars.points - 1] - vars.x[0]).abs();
        for k in 0..vars.points - 1 {
            if mark[k] {
                mean = half * (vars.x[k] + vars.x[k + 1]);
                if !((vars.x[k] < mean && mean < vars.x[k + 1]) || (vars.x[k + 1] < mean && mean < vars.x[k])) {
                    counted += 1;
                }
            }
        }
        *error = counted > 0;
        if *error {
            if text > 0 {
                println!("{}ERROR.  SOME INTERVALS IN THE GRID ARE TOO SHORT.\n\
                THE NEW GRID WOULD NOT BE ORDERED.", id);
            }
            return;
        }

        // add the new points, interpolate x and the bounds.
        new = total;
        for old in (1..vars.points - 1).rev() {
            // Copy right boundary
            vars.x[new] = vars.x[old];
            for j in 0..vars.comps {
                u[j * total + new] = u[j * vars.points + old];
            }
            new -= 1;
            // Interpolate solution and location
            if mark[old - 1] {
                vars.x[new] = half * (vars.x[old] + vars.x[old - 1]);
                for j in 0..vars.comps {
                    u[j * total + new] = half * (u[j * vars.points + old] + u[j * vars.points + old - 1]);
                }
                new -= 1;
            }
        }

        // mark the new points.
        new = total;
        for old in (1..vars.points - 1).rev() {
            mark[new] = false;
            new -= 1;
            if mark[old - 1] {
                mark[new] = true;
                new -= 1;
            }
        }
        mark[new] = false;

        // update the number of points.
        vars.points = total;

        // Allow the user to update the solution.
        buffer.copy_from_slice(u);
        this.update_grid(error, vars, buffer);
        u.copy_from_slice(buffer);
        if *error {
            if levelm > 0 && text > 0 {
                println!("{}ERROR.  USER-DEFINED SOLUTION UPDATE ON NEW GRID FAILED.", id);
            }
            return;
        }
    }

    // ***** epilogue. *****
    // print summary
    if levelm > 0 && text > 0 {
        temp1 = ratio1.iter().take(former - 1).fold(0.0, |acc, &x| acc.max(x));
        temp2 = ratio2.iter().skip(1).take(former - 2).fold(0.0, |acc, &x| acc.max(x));
        if signif == 0 {
            println!("{}SUCCESS.  THE GRID IS ADEQUATE BECAUSE ALL ACTIVE COMPONENTS ARE INSIGNIFICANT.", id);
        } else {
            println!(
                "{}{:>10.3}  {:>10.3}  {:>10.2}  {:>10.2}",
                id, temp1, temp2, toler1, toler2
            );
            if most == 0 {
                println!("{}SUCCESS.  THE GRID IS ADEQUATE.",


            }
        }
    }
}   





impl TwoPntBVProblem {
    pub fn refine(
        &mut self,
        error: &mut bool,
        setup: &TwoPntSolverSetup,
        text: i32,
        buffer: &mut [f64],
        vars: &mut TwoPntBVPDomain,
        mark: &mut [bool],
        newx: &mut bool,
        ratio: &mut [f64; 2],
        ratio1: &mut [f64],
        ratio2: &mut [f64],
        success: &mut bool,
        u: &mut [f64],
        vary1: &mut [i32],
        vary2: &mut [i32],
        weight: &mut [i32],
    ) {
        const ID: &str = "REFINE:  ";
        let zero = 0.0;
        let one = 1.0;
        let half = 0.5;

        // Initialization: turn off all completion status flags.
        *error = false;
        *newx = false;
        *success = false;

        // Levelm printing.
        if setup.levelm > 0 && text > 0 {
            println!("{}SELECT A GRID.", ID);
        }

        // Check the arguments.
        vars.check_on_grid_update(error, ID, text, setup.ipadd);
        if *error {
            return;
        }

        // Check there is at least one variable that affects grid adaption
        let counted = vars.active.iter().filter(|&&x| x).count();
        *error = counted < 1;
        if *error {
            if text > 0 {
                println!("{}ERROR.  THERE ARE NO ACTIVE COMPONENTS.", ID);
            }
            return;
        }

        *error = !(setup.toler0 >= zero);
        if *error {
            if text > 0 {
                println!("{}ERROR.  THE BOUNDS ON MAGNITUDE AND RELATIVE CHANGE OF MAGNITUDE FOR INSIGNIFICANT COMPONENTS MUST BE POSITIVE.\n{:10.2e}  TOLER0, SIGNIFICANCE LEVEL", ID, setup.toler0);
            }
            return;
        }

        // Check tolerances in [0,1]
        *error = !(zero <= setup.toler1 && setup.toler1 <= one && zero <= setup.toler2 && setup.toler2 <= one);
        if *error {
            if text > 0 {
                println!("{}ERROR.  THE BOUNDS ON RELATIVE CHANGES IN MAGNITUDE AND ANGLE MUST LIE BETWEEN 0 AND 1.\n{:10.2e}  TOLER1\n{:10.2e}  TOLER2", ID, setup.toler1, setup.toler2);
            }
            return;
        }

        // Check monotonic
        let counted = vars.x.windows(2).filter(|w| w[0] < w[1]).count();
        *error = !(counted == 0 || counted == vars.points - 1);
        if *error {
            if text > 0 {
                println!("{}ERROR.  THE GRID IS NOT ORDERED.", ID);
            }
            return;
        }

        // Initialize variables
        let mut act = 0;
        let mut signif = 0;
        for k in 0..vars.points {
            mark[k] = false;
            ratio1[k] = zero;
            ratio2[k] = zero;
            vary1[k] = 0;
            vary2[k] = 0;
        }

        // Loop over components
        for j in 0..vars.comps {
            if !vars.active[j] {
                continue;
            }
            act += 1;

            // Find range and maximum magnitude of this component
            let mut lower = u[j * vars.points];
            let mut upper = u[j * vars.points];
            for k in 1..vars.points {
                lower = lower.min(u[j * vars.points + k]);
                upper = upper.max(u[j * vars.points + k]);
            }
            let range = upper - lower;
            let maxmag = lower.abs().max(upper.abs());

            // Decide whether the component is significant
            if !(range.abs() > setup.toler0 * one.max(maxmag)) {
                continue;
            }

            // This is a significant component
            signif += 1;

            // Check component's change in each interval
            for k in 0..vars.points - 1 {
                let differ = (u[j * vars.points + k + 1] - u[j * vars.points + k]).abs();
                if zero < range {
                    ratio1[k] = ratio1[k].max(differ / range);
                }
                if setup.toler1 * range < differ {
                    vary1[k] += 1;
                }
            }

            // Find the global change of the component's derivative
            let mut temp = grad(u, &vars.x, j, 1);
            let mut lower = temp;
            let mut upper = temp;
            for k in 2..vars.points - 1 {
                temp = grad(u, &vars.x, j, k);
                lower = lower.min(temp);
                upper = upper.max(temp);
            }
            let range = upper - lower;

            // Check derivative's change at each interior point
            let mut right = grad(u, &vars.x, j, 1);
            for k in 2..vars.points - 1 {
                let left = right;
                right = grad(u, &vars.x, j, k);
                let differ = (left - right).abs();
                if zero < range {
                    ratio2[k] = ratio2[k].max(differ / range);
                }
                if setup.toler2 * range < differ {
                    vary2[k] += 1;
                }
            }
        }

        // Save the maximum ratios
        ratio[0] = ratio1.iter().take(vars.points - 1).fold(zero, |acc, &x| acc.max(x));
        ratio[1] = ratio2.iter().skip(1).take(vars.points - 2).fold(zero, |acc, &x| acc.max(x));

        // Select the intervals to halve
        let mut most = 0;
        for k in 0..vars.points - 1 {
            weight[k] = vary1[k];
            if k > 0 {
                weight[k] += vary2[k];
            }
            if k < vars.points - 2 {
                weight[k] += vary2[k + 1];
            }
            if weight[k] > 0 {
                most += 1;
            }
        }

        // Sort the weights using interchange sort
        for k in 0..vars.points - 1 {
            for j in k + 1..vars.points - 1 {
                if weight[j] > weight[k] {
                    let temp = weight[j];
                    weight[j] = weight[k];
                    weight[k] = temp;
                }
            }
            if weight[k] == 0 {
                break;
            }
        }

        // Find the least weight of intervals to halve
        let more = most.min(setup.ipadd).min(vars.pmax - vars.points);
        let least = if more > 0 {
            weight[more - 1]
        } else {
            1 + weight[0]
        };

        // Reconstruct the weights
        for k in 0..vars.points - 1 {
            weight[k] = vary1[k];
            if k > 0 {
                weight[k] += vary2[k];
            }
            if k < vars.points - 2 {
                weight[k] += vary2[k + 1];
            }
        }

        // Mark the intervals to halve
        let mut counted = 0;
        for k in 0..vars.points - 1 {
            if counted < more && least <= weight[k] {
                counted += 1;
                mark[k] = true;
            }
        }

        // Halve the intervals, if any
        let total = vars.points + counted;
        let former = vars.points;

        if counted > 0 {
            // Check that the new points are not degenerate
            let mut degenerate_count = 0;
            let length = (vars.x[vars.points - 1] - vars.x[0]).abs();
            for k in 0..vars.points - 1 {
                if mark[k] {
                    let mean = half * (vars.x[k] + vars.x[k + 1]);
                    if !((vars.x[k] < mean && mean < vars.x[k + 1]) || (vars.x[k + 1] < mean && mean < vars.x[k])) {
                        degenerate_count += 1;
                    }
                }
            }
            *error = degenerate_count > 0;
            if *error {
                if text > 0 {
                    println!("{}ERROR.  SOME INTERVALS IN THE GRID ARE TOO SHORT.\nTHE NEW GRID WOULD NOT BE ORDERED.", ID);
                }
                return;
            }

            // Add the new points, interpolate x and the bounds
            let mut new = total - 1;
            for old in (1..vars.points).rev() {
                // Copy right boundary
                vars.x[new] = vars.x[old];
                for j in 0..vars.comps {
                    u[j * total + new] = u[j * vars.points + old];
                }
                new -= 1;
                // Interpolate solution and location
                if mark[old - 1] {
                    vars.x[new] = half * (vars.x[old] + vars.x[old - 1]);
                    for j in 0..vars.comps {
                        u[j * total + new] = half * (u[j * vars.points + old] + u[j * vars.points + old - 1]);
                    }
                    new -= 1;
                }
            }

            // Mark the new points
            new = total - 1;
            for old in (1..vars.points).rev() {
                mark[new] = false;
                new -= 1;
                if mark[old - 1] {
                    mark[new] = true;
                    new -= 1;
                }
            }
            mark[new] = false;

            // Update the number of points
            vars.points = total;

            // Allow the user to update the solution
            buffer.copy_from_slice(&u[..vars.comps * vars.points]);
            self.update_grid(error, vars, buffer);
            u[..vars.comps * vars.points].copy_from_slice(buffer);
            if *error {
                if setup.levelm > 0 && text > 0 {
                    println!("{}ERROR.  USER-DEFINED SOLUTION UPDATE ON NEW GRID FAILED.", ID);
                }
                return;
            }
        }

        // Epilogue
        if setup.levelm > 0 && text > 0 {
            let temp1 = ratio1.iter().take(former - 1).fold(zero, |acc, &x| acc.max(x));
            let temp2 = ratio2.iter().skip(1).take(former - 2).fold(zero, |acc, &x| acc.max(x));
            if signif == 0 {
                println!("{}SUCCESS.  THE GRID IS ADEQUATE BECAUSE ALL ACTIVE COMPONENTS ARE INSIGNIFICANT.", ID);
                return;
            }
            println!("{}SUCCESS.  THE GRID HAS BEEN REFINED {} TIMES.", ID, signif);
            println!("  THE NEW GRID HAS {} POINTS.", vars.points);
            println!("  THE MAXIMUM RATIOS ARE {} AND {}.", temp1, temp2);
        }
    }
}






use std::cmp::{max, min};
use std::f64::EPSILON;

const ZERO: f64 = 0.0;
const ONE: f64 = 1.0;
const HALF: f64 = 0.5;

struct TwoPntBVProblem {
    // Define the fields as per the Fortran code
}

struct TwoPntSolverSetup {
    leveld: i32,
    levelm: i32,
    ipadd: i32,
    toler0: f64,
    toler1: f64,
    toler2: f64,
}

struct TwoPntBVPDomain {
    points: usize,
    pmax: usize,
    comps: usize,
    active: Vec<bool>,
    _x: Vec<f64>,
}

fn grad(u: &Vec<Vec<f64>>, _x: &Vec<f64>, comp: usize, point: usize) -> f64 {
    (u[comp][point + 1] - u[comp][point]) / (_x[point + 1] - _x[point])
}

fn twcopy(n: usize, from: &Vec<f64>, to: &mut Vec<f64>) {
    for i in 0..n {
        to[i] = from[i];
    }
}

fn refine(
    this: &mut TwoPntBVProblem,
    error: &mut bool,
    setup: &TwoPntSolverSetup,
    text: i32,
    buffer: &mut Vec<f64>,
    vars: &mut TwoPntBVPDomain,
    mark: &mut Vec<bool>,
    newx: &mut bool,
    ratio: &mut [f64; 2],
    ratio1: &mut Vec<f64>,
    ratio2: &mut Vec<f64>,
    success: &mut bool,
    u: &mut Vec<Vec<f64>>,
    vary1: &mut Vec<i32>,
    vary2: &mut Vec<i32>,
    weight: &mut Vec<i32>,
) {
    let id = "REFINE:  ";
    let mut word = String::new();
    let mut differ;
    let mut left;
    let mut length;
    let mut lower;
    let mut maxmag;
    let mut mean;
    let mut range;
    let mut right;
    let mut temp;
    let mut temp1;
    let mut temp2;
    let mut upper;
    let mut act;
    let mut counted;
    let mut former;
    let mut itemp;
    let mut j;
    let mut k;
    let mut least;
    let mut more;
    let mut most;
    let mut new;
    let mut old;
    let mut signif;
    let mut total;

    *error = false;
    *newx = false;
    *success = false;

    if setup.levelm > 0 && text > 0 {
        println!("{}", id);
    }

    // Check the arguments
    // call vars%check_onGridUpdate(error, id, text, padd)
    if *error {
        return;
    }

    // Check there is at least one variable that affects grid adaption
    counted = vars.active.iter().filter(|&&x| x).count();
    *error = counted < 1;
    if *error {
        if text > 0 {
            println!("{}: No active components", id);
        }
        return;
    }

    *error = setup.toler0 < ZERO;
    if *error {
        if text > 0 {
            println!("{}: Invalid toler0: {}", id, setup.toler0);
        }
        return;
    }

    *error = !(ZERO <= setup.toler1 && setup.toler1 <= ONE && ZERO <= setup.toler2 && setup.toler2 <= ONE);
    if *error {
        if text > 0 {
            println!("{}: Invalid toler1 or toler2: {}, {}", id, setup.toler1, setup.toler2);
        }
        return;
    }

    counted = vars._x.iter().zip(vars._x.iter().skip(1)).filter(|(&a, &b)| a < b).count();
    *error = !(counted == 0 || counted == vars.points - 1);
    if *error {
        if text > 0 {
            println!("{}: Grid is not ordered", id);
        }
        return;
    }

    act = 0;
    signif = 0;
    mark.fill(false);
    ratio1.fill(ZERO);
    ratio2.fill(ZERO);
    vary1.fill(0);
    vary2.fill(0);

    for j in 0..vars.comps {
        if !vars.active[j] {
            continue;
        }

        act += 1;

        lower = u[j][0];
        upper = u[j][0];
        for k in 1..vars.points {
            lower = lower.min(u[j][k]);
            upper = upper.max(u[j][k]);
        }
        range = upper - lower;
        maxmag = lower.abs().max(upper.abs());

        if !(range.abs() > setup.toler0 * max(ONE, maxmag)) {
            continue;
        }

        signif += 1;

        for k in 0..vars.points - 1 {
            differ = (u[j][k + 1] - u[j][k]).abs();
            if range > ZERO {
                ratio1[k] = ratio1[k].max(differ / range);
            }
            if setup.toler1 * range < differ {
                vary1[k] += 1;
            }
        }

        temp = grad(&u, &vars._x, j, 0);
        lower = temp;
        upper = temp;
        for k in 1..vars.points - 1 {
            temp = grad(&u, &vars._x, j, k);
            lower = lower.min(temp);
            upper = upper.max(temp);
        }
        range = upper - lower;

        right = grad(&u, &vars._x, j, 0);
        for k in 1..vars.points - 1 {
            left = right;
            right = grad(&u, &vars._x, j, k);
            differ = (left - right).abs();
            if range > ZERO {
                ratio2[k] = ratio2[k].max(differ / range);
            }
            if setup.toler2 * range < differ {
                vary2[k] += 1;
            }
        }
    }

    ratio[0] = ratio1.iter().take(vars.points - 1).cloned().fold(ZERO, f64::max);
    ratio[1] = ratio2.iter().skip(1).take(vars.points - 2).cloned().fold(ZERO, f64::max);

    most = 0;
    for k in 0..vars.points - 1 {
        weight[k] = vary1[k];
        if k > 0 {
            weight[k] += vary2[k];
        }
        if k < vars.points - 1 {
            weight[k] += vary2[k + 1];
        }
        if weight[k] > 0 {
            most += 1;
        }
    }

    for k in 0..vars.points - 1 {
        for j in k + 1..vars.points - 1 {
            if weight[j] > weight[k] {
                itemp = weight[j];
                weight[j] = weight[k];
                weight[k] = itemp;
            }
        }
        if weight[k] == 0 {
            break;
        }
    }

    more = max(0, min(most, setup.ipadd as usize, vars.pmax - vars.points));
    least = if more > 0 {
        weight[more - 1]
    } else {
        weight[0] + 1
    };

    for k in 0..vars.points - 1 {
        weight[k] = vary1[k];
        if k > 0 {
            weight[k] += vary2[k];
        }
        if k < vars.points - 1 {
            weight[k] += vary2[k + 1];
        }
    }

    counted = 0;
    for k in 0..vars.points - 1 {
        if counted < more && least <= weight[k] {
            counted += 1;
            mark[k] = true;
        }
    }

    more = counted;

    total = vars.points + more;
    former = vars.points;

    if more > 0 {
        counted = 0;
        length = (vars._x[vars.points - 1] - vars._x[0]).abs();
        for k in 0..vars.points - 1 {
            if mark[k] {
                mean = HALF * (vars._x[k] + vars._x[k + 1]);
                if !((vars._x[k] < mean && mean < vars._x[k + 1]) || (vars._x[k + 1] < mean && mean < vars._x[k])) {
                    counted += 1;
                }
            }
        }
        *error = counted > 0;
        if *error {
            if text > 0 {
                println!("{}: Some intervals in the grid are too short", id);
            }
            return;
        }

        new = total;
        for old in (1..vars.points).rev() {
            vars._x[new - 1] = vars._x[old];
            for j in 0..vars.comps {
                u[j][new - 1] = u[j][old];
            }
            new -= 1;
            if mark[old - 1] {
                vars._x[new - 1] = HALF * (vars._x[old] + vars._x[old - 1]);
                for j in 0..vars.comps {
                    u[j][new - 1] = HALF * (u[j][old] + u[j][old - 1]);
                }
                new -= 1;
            }
        }

        new = total;
        for old in (1..vars.points).rev() {
            mark[new - 1] = false;
            new -= 1;
            if mark[old - 1] {
                mark[new - 1] = true;
                new -= 1;
            }
        }
        mark[new - 1] = false;

        vars.points = total;

        twcopy(vars.comps * vars.points, &u.iter().flatten().cloned().collect(), buffer);
        // call this%update_grid(error, vars, buffer)
        twcopy(vars.comps * vars.points, buffer, &mut u.iter_mut().flatten().collect());
        if *error {
            if setup.levelm > 0 && text > 0 {
                println!("{}: User-defined solution update on new grid failed", id);
            }
            return;
        }
    }

    if setup.levelm > 0 && text > 0 {
        temp1 = ratio1.iter().take(former - 1).cloned().fold(ZERO, f64::max);
        temp2 = ratio2.iter().skip(1).take(former - 2).cloned().fold(ZERO, f64::max);

        if signif == 0 {
            println!("{}: The grid is adequate because all active components are insignificant", id);
        } else {
            println!("{}: Ratios: Actual: {:.3}, {:.3}, Desired: {:.3}, {:.3}", id, temp1, temp2, setup.toler1, setup.toler2);
            if most == 0 {
                println!("{}: The grid is adequate", id);
            } else if more == 0 {
                println!("{}: More points are needed but none can be added", id);
            } else {
                println!("{}: The new grid (* marks new points):", id);
                println!("{}: Index, Grid Point, Largest Ratios and Number Too Large", id);
                old = 0;
                for k in 0..vars.points {
                    if !mark[k] {
                        old += 1;
                        if k > 0 {
                            if vary1[old - 1] != 0 {
                                word = format!("{:.2}, {}", ratio1[old - 1], vary1[old - 1]);
                            } else {
                                word = format!("{:.2}", ratio1[old - 1]);
                            }
                            if mark[k - 1] {
                                println!("{}: {}*, {:.9}, {}", id, k - 1, vars._x[k - 1], word);
                            } else {
                                println!("{}: {}", id, word);
                            }
                        }
                        if k > 0 && k < vars.points {
                            if vary2[old] != 0 {
                                word = format!("{:.2}, {}", ratio2[old], vary2[old]);
                            } else {
                                word = format!("{:.2}", ratio2[old]);
                            }
                            println!("{}: {}, {:.9}, {}", id, k, vars._x[k], word);
                        } else {
                            println!("{}: {}, {:.9}", id, k, vars._x[k]);
                        }
                    }
                }
            }
        }

        if setup.leveld > 0 && more > 0 {
            println!("{}: The solution guess for the new grid:", id);
            twcopy(vars.comps * vars.points, &u.iter().flatten().cloned().collect(), buffer);
            // call this%show(error, text, buffer, vars, true)
            if *error {
                return;
            }
        }
    }

    *newx = more > 0;
    *success = most == 0;
}




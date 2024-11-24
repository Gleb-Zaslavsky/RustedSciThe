use nalgebra::{DMatrix, DVector, };
use std::collections::HashMap;



/*
Some math considerations...
1. truncation error analysis
Consider a forkward finite difference approximation of the first-order derivative u
[Dy]_n = (y_(n+1)-y_n)/dt (1)
Here, y_n means the value of some function u(t) at a point tn, and [Dy]_n is the discrete derivative
of y(t) at t = t_n. The discrete derivative computed by a finite difference is not exactly equal to
the derivative y'(t_n) at t = t_n. The error in the approximation is
R_n = [Dy]_n -  y'(t_n) (2)
The common way of calculating Rn is to
1. expand y(t) in a Taylor series around the point where the derivative is evaluated, here t_n,
2. insert this Taylor series in (2), and
3. collect terms that cancel and simplify the expression.
 - The Taylor series of y_n at t_n is simply y_n=y(t_n),
 - The Taylor series of y_(n+1) at t_n is y_(n+1) =y(t_n + dt)= y(t_n) + y'(t_n)*dt + y''(t_n)*dt^2/2 + ... 
 Inserting the Taylor series above in the left-hand side of (2) gives
 R_n = [Dy]_n -  y'(t_n) = (y_(n+1)-y_n)/dt - y'(t_n) = (y_n + y'(t_n)*dt + y''(t_n)*dt^2/2 -y_n )/dt - y'(t_n) =   y''(t_n)*dt/2 
so for  forkward finite difference truncation error is R_n =   y''(t_n)*dt/2 
For backwarw finite difference [Dy]_n = (y_n - y_(n-1))/d truncation error is R_n =  - y''(t_n)*dt/2 (because y_(n-1) =y(t_n - dt)= y(t_n) - y'(t_n)*dt + y''(t_n)*dt^2/2 + ...  )
For the central difference approximation,

 [Dy]_n = (y_(n+0.5)-y_(n-0.5))/dt (3)
we write
R_n = [Dy]_n -  y'(t_n)
 The Taylor series of y_(n+0.5)  at t_n is y_(n+0.5) =y(t_n + 0.5dt)= y(t_n) +0.5 y'(t_n)*dt + y''(t_n)*(0.5*dt)^2/2 + y'''(t_n)*(0.5*dt)^3/6
  The Taylor series of y_(n+0.5)  at t_n is y_(n-0.5) =y(t_n - 0.5dt)= y(t_n) - 0.5 y'(t_n)*dt + y''(t_n)*(0.5*dt)^2/2 - y'''(t_n)*(0.5*dt)^3/6
  y_(n+0.5) - y_(n+0.5)  =  y'(t_n)*dt + 2* y'''(t_n)*(0.5*dt)^3/6 =  y'(t_n)*dt +  y'''(t_n)*(dt)^3/24 => [Dy]_n = y'(t_n) +  y'''(t_n)*(dt)^2/24 =>
  R_n =  y'''(t_n)*(dt)^2/24
dimension of R_n is always just the same as the dimension of derivative
2. On the choise of mesh 
"We have found that starting the itration on a coarse mesh has several important advntages. One is that the Newton iteration is more likely to 
converge on a coarse mesh than on a fine mesh. Moreover, the number of variables is small on a coarse mesh and thus the cost per iteration is 
relatively small. Since the iteration begins from a user-specfied “guess” at the solution, it is likly that many iterations will be required. 
Ultimately, of course, to be accurate, the solution must be obtained on a fine mesh. However, as the solution is computed on each successively finer 
mesh, the starting estimates are better, since they come from the converged solution on the previous coarse mesh. In general, the solution on one
mesh lies within the domain of convergence of Newton’s method on the next finer mesh.Thus, even though the  cost per iteration is increasing, the 
number of required iterations is decreasing. The adaptve placement of the mesh points to form the finer meshes is done in such a 
way that the total number of mesh points needed to represent the solution accurately is minimized
 " Chemkin Theory Manual p.263 
*/


pub fn easiest_grid_refinement(y_DMatrix:&DMatrix<f64>, x_mesh:&DVector<f64>, tolerance:f64, safety_param:f64 ) -> (Vec<f64>,DMatrix<f64>, usize) {
    let mut h:Vec<f64> = Vec::new();
    let mut new_initial_guess:Vec<f64> = Vec::new();
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid:Vec<f64> = Vec::new();
    let mut mark:DVector<i8> = DVector::zeros(x_mesh.len());
    //println!("{} rows", n_rows);
    // each row is the solution of the ODE at a points in the grid
    for (j, y) in y_DMatrix.row_iter().enumerate() {
                
                // iterate through the the row (solution for one of the unknown variables) and find the corresponding truncation error that is 
                // larger than the tolerance/safety_param. If it is, the point is marked for refinement.  else not.
                // Also, we are not refining the first and last points (i==0 and i==-1) because they are not at the boundary.
                for i in 0..x_mesh.len() {
                    new_initial_guess.push(y[i]);// first we just copy element to new_initial_guess
                    if x_mesh.len()-1>i && i>0 {
                    let h_i  = x_mesh[i+1]-x_mesh[i];
                    h.push(h_i); 
                    // truncation error    
                    let tau_i = (1.0/(2.0*h_i  )* (y[i+1]-2.0*y[i]+y[i-1])  ).abs(); 
                    if tau_i>tolerance/safety_param {
                    mark[i] = 1;
                    // interpolate between y[i] and y[i+1] and add new point to new_initial_guess
                    let dy_i = y[i+1]-y[i];
                    let y_new = y[i] + dy_i/2.0;
                    new_initial_guess.push(y_new);
                    } // if not mark element remains 0
                  
                    }//i>0      
                    //for i==0 and i==-1 mark elements remain 0
               
                }// for i in 0..x_mesh.len()
                println!("\n \n for row {} mark: {:?} len {} \n \n",j, mark, mark.len());

            } 
          
            for i in 0..x_mesh.len() {
                        if mark[i]==1 {
                            new_grid.push(x_mesh[i]);// old points included
                            let h_i = x_mesh[i+1]-x_mesh[i];
                            let x_new = x_mesh[i] + h_i/2.0;
                            new_grid.push(x_new);            
                        }
                        
                        else {new_grid.push(x_mesh[i])}
            
            }  

   
log::info!("\n \n new_grid: {:?}",new_grid);  
log::info!("of length {}", new_grid.len());   
log::info!("\n \n new_initial_guess: {:?} of length {}", new_initial_guess, new_initial_guess.len());
log::info!("will be converted to DMatrix wirh number of columns {} and rows {}", new_grid.len(), n_rows);
assert_eq!(new_initial_guess.len(),new_grid.len()*n_rows);

let number_of_nonzero_keys = mark.iter().sum::<i8>() as usize;
log::info!("number of nonzero keys {}", number_of_nonzero_keys);
let  new_initial_guess:DMatrix<f64> =  DMatrix::from_vec(n_rows,new_grid.len(), new_initial_guess);

(new_grid, new_initial_guess,  number_of_nonzero_keys)

}


/*
Saurces:
1) ON  A  DIFFERENTIAL  EQUATION  OF  BOUNDARY  LAYER  TYPE 
By CARL  E.  PEARSON,
p. 138 
2) 
eng
New mesh points are now inserted between any pair of adjacent mesh points-say x_i and x_i+1  for which (y_i+1 - y_i).abs() exceeds a predetermined 
limit delta; the number of such mesh points inserted (uniformly)  between x_i and x_i+1 is approximately equal to (y_i+1 - y_i).abs()/delta.  
Discrete Equations  are then solved again, new  mesh points inserted, and so  on; the process continues iteratively until (y_i+1 - y_i).abs()< delta 
everywhere. The value of delta is  adjusted during the computation, so  as to always bear a fixed ratio  (typically 1e-3)  to the computed value of 
 (max_i {y_i)}  - min_i {y_i} }.   Since the insertion of new  mesh points may result in a locally abrupt change in mesh interval size,  with some  
 consequent  loss  in  the accuracy  with which Discrete Eq. approximates continuous  Eq.,  a  smoothing  process  is  carried  out  prior  to each  
 new Gaussian elimination sweep.  This smoothing process simply consists in replacing each mesh point  x_i  by a new mesh point  x_i' = 0.5(x_i+ x_i+1)

 Новые точки сетки теперь вставляются между любой парой соседних точек сетки, скажем, x_i и x_i+1, для которых (y_i+1 - y_i).abs() превышает 
 определенный предел delta; количество таких точек сетки, вставленных (равномерно) между x_i и x_i+1, приблизительно равно 
 (y_i+1 - y_i).abs()/delta. Затем снова решаются дискретные уравнения, вставляются новые точки сетки и так далее; процесс продолжается итеративно 
 до тех пор, пока (y_i+1 - y_i).abs()< delta везде. Значение delta корректируется во время вычисления, чтобы всегда иметь фиксированное отношение
  (обычно 1e-3) к вычисленному значению (max_i {y_i)} - min_i {y_i} }. Поскольку вставка новых точек сетки может привести к локальному резкому 
  изменению размера интервала сетки с некоторой последующей потерей точности, с которой дискретные уравнения. аппроксимируют непрерывные, перед каждым новым 
  гауссовым исключением выполняется процесс сглаживания. Этот процесс сглаживания заключается просто в замене каждой точки сетки x_i новой точкой 
  сетки x_i' = 0.5(x_i+ x_i+1)

 */

 pub fn pearson_grid_refinement(y_DMatrix:&DMatrix<f64>, x_mesh:&DVector<f64>,d:f64 ) -> (Vec<f64>,DMatrix<f64>, usize) {
    let mut h:Vec<f64> = Vec::new();
    let mut new_initial_guess:Vec<f64> = Vec::new();
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid:Vec<f64> = Vec::new();
    // hashmap key: in what position insert points, value: how many points to insert
     // mark[i] = how many points to insert in i-th position
    let mut mark:HashMap<usize, i32> =  HashMap::new();


    //println!("{} rows", n_rows);
    // each row is the solution of the ODE at a points in the grid
    for (j, y) in y_DMatrix.row_iter().enumerate() {
                
                let y_j_max = y.max();
                let y_j_min = y.min();
                let delta = d*(y_j_max - y_j_min);
                for i in 0..x_mesh.len() {
                    mark.insert(i, 0); // default value 0 - no point inserted
                    new_initial_guess.push(y[i]);// first we just copy element to new_initial_guess
                    if x_mesh.len()-1>i && i>0 {
                    let h_i  = x_mesh[i+1]-x_mesh[i];
                    h.push(h_i); 
                    let dy_i = y[i+1]-y[i];
                    // truncation error    
                    let tau_i = dy_i.abs(); 
                    if tau_i>delta {
                    // how many new points should be added
                    let N =   (tau_i/delta) as i32;
                    mark.insert(i, N); // mark[i] = how many points to insert in i-th position
                    
                    for k in 0..N {        
                        // add new point to new_initial_guess
                        let y_new = y[i] + dy_i*(k as f64 )/N as f64;
                        new_initial_guess.push(y_new);
                    } 
          
                    } // if not mark element remains 0
                    else { 
                        let y_new = y[i] + dy_i/2.0;
                        new_initial_guess.push(y_new);
                    }
                  
                    }//i>0      
                    //for i==0 and i==-1 mark elements remain 0
               
                }// for i in 0..x_mesh.len()
                // find keys corresponding to non-zero values in the HashMap
                let non_zero_keys: Vec<usize> = mark.iter()
                .filter(|(_, &value)| value != 0)
                .map(|(key, _)| *key)
                .collect();
                println!("\n \n for row {} found intervals to be refined: {:?} of length {} \n \n",j, non_zero_keys, non_zero_keys.len());

            } 
          
            for i in 0..x_mesh.len()-1 {
                        if mark.get(&i).unwrap() != &0 {
                            new_grid.push(x_mesh[i]);// old points included
                            let N = *mark.get(&i).unwrap();
                            let h_i = x_mesh[i+1]-x_mesh[i];
                            for k in 0..N {    
                            let x_new = x_mesh[i] + h_i*(k as f64 )/(N as f64);
                            new_grid.push(x_new);   }         
                        }
                        else {
                            let h_i = x_mesh[i+1]-x_mesh[i];
                            let x_new = x_mesh[i] + h_i/2.0;
                            new_grid.push(x_mesh[i]);
                            new_grid.push(x_new) ;
                        }            
            }  

   
log::info!("\n \n new_grid: {:?}",new_grid);  
log::info!("of length {}", new_grid.len());   
log::info!("\n \n new_initial_guess: {:?} of length {}", new_initial_guess, new_initial_guess.len());
log::info!("will be converted to DMatrix wirh number of columns {} and rows {}", new_grid.len(), n_rows);
assert_eq!(new_initial_guess.len(),new_grid.len()*n_rows, );

let number_of_nonzero_keys = mark.iter().map(|(x, y)| if *y != 0 {1} else {0}).sum::<i32>() as usize;
log::info!("number of nonzero keys {}", number_of_nonzero_keys);
let  new_initial_guess:DMatrix<f64> =  DMatrix::from_vec(n_rows,new_grid.len(), new_initial_guess);

(new_grid, new_initial_guess,  number_of_nonzero_keys)

}

/*
Saurces
A  HYBRID NEWTON/TIME-INTEGRATION  PROCEDURE FOR THE 
SOLUTION OF STEADY, LAMINAR,  ONE-DIMENSIONAL,  PREMIXED 
FLAMES by JOSEPH F. GRCAR, ROBERT J.  KEE, MITCHELL D. SMOOKE and JAMES A. MILLER 

The  starting  estimate  for  the  dependent variable  vector  y  on  a  new,  finer  mesh  is determined by a  linear interpolation of the old 
coarse  mesh  solution.  After  obtaining a  converged  solution on the  new mesh,  the adapta-tion  procedure  is  performed  once  again.  A 
sequence  of  solutions  on  successively  finer meshes  is  computed  until  the  inequalities in Eqs. (1) and (2) (see code below) are satisfied between all mesh 
points.


olution  of  Burner-Stabilized  Premixed  Laminar  Flames 
by  Boundary  Value  Methods 
by MITCHELL  D.  SMOOKE 


*/
pub fn grcar_smooke_grid_refinement(y_DMatrix:&DMatrix<f64>, x_mesh:&DVector<f64>,d:f64, g:f64 ) -> (Vec<f64>,DMatrix<f64>, usize) {
    let mut h:Vec<f64> = Vec::new();
    let mut new_initial_guess:Vec<f64> = Vec::new();
    let (n_rows, _) = y_DMatrix.shape();
    let mut new_grid:Vec<f64> = Vec::new();
    // hashmap key: in what position insert points, value: how many points to insert
     // mark[i] = how many points to insert in i-th position
    let mut mark:HashMap<usize, i32> =  HashMap::new();


    //println!("{} rows", n_rows);
    // each row is the solution of the ODE at a points in the grid
    for (j, y) in y_DMatrix.row_iter().enumerate() {
                
                let y_j_max = y.max();
                let y_j_min = y.min();
                let delta = d*(y_j_max - y_j_min);
                let mut list_dy_dx_i =Vec::new();
       
                for i in 0..x_mesh.len()-1 {
                    let dy_i = y[i+1] - y[i];
                    let h_i  = x_mesh[i+1]-x_mesh[i];
                    let dy_dx_i = dy_i/h_i;
                    list_dy_dx_i.push(dy_dx_i);

                }
                let  list_dy_dx_i_min = list_dy_dx_i.iter().cloned().min_by(|a, b| a.total_cmp(b)).unwrap();
                let  list_dy_dx_i_max = list_dy_dx_i.iter().cloned().max_by(|a, b| a.total_cmp(b)).unwrap();
                let derivative_range = (list_dy_dx_i_max - list_dy_dx_i_min).abs();
                let gamma = g*derivative_range;
                for i in 0..x_mesh.len() {
                    mark.insert(i, 0); // default value 0 - no point inserted
                    new_initial_guess.push(y[i]);// first we just copy element to new_initial_guess
                    if x_mesh.len()-1>i && i>0 {
                    let dy_i = y[i+1] - y[i];
                    let h_i  = x_mesh[i+1]-x_mesh[i];
                    let dy_dx_i = dy_i/h_i;

                    let dy_i_min_1 = y[i] - y[i-1];
                    let h_i_min_1 = x_mesh[i]-x_mesh[i-1];
                    let dy_dx_i_min_1 = dy_i_min_1/h_i_min_1;

                    h.push(h_i); 
                    let eta_i = (dy_dx_i - dy_dx_i_min_1).abs();  
                    let tau_i = (y[i]- y[i-1]).abs(); 
                    // eq 1, eq 2 
                    if tau_i>delta || eta_i>gamma  {
                    // how many new points should be added
                    let N =   (tau_i/delta) as i32;
                    mark.insert(i, N); // mark[i] = how many points to insert in i-th position
                    let dy_i = y[i+1]-y[i];
                    for k in 0..N {        
                        // add new point to new_initial_guess
                        let y_new = y[i] + dy_i*(k as f64 )/N as f64;
                        new_initial_guess.push(y_new);
                    }
          
                    } // if not mark element remains 0
                    else { 
                        let y_new = y[i] + dy_i/2.0;
                        new_initial_guess.push(y_new);
                    }
                  
                    }//i>0      
                    //for i==0 and i==-1 mark elements remain 0
               
                }// for i in 0..x_mesh.len()
                // find keys corresponding to non-zero values in the HashMap
                let non_zero_keys: Vec<usize> = mark.iter()
                .filter(|(_, &value)| value != 0)
                .map(|(key, _)| *key)
                .collect();
                println!("\n \n for row {} found intervals to be refined: {:?} of length {} \n \n",j, non_zero_keys, non_zero_keys.len());

            } 
          
            for i in 0..x_mesh.len()-1 {
                        if mark.get(&i).unwrap() != &0 {
                            new_grid.push(x_mesh[i]);// old points included
                            let N = *mark.get(&i).unwrap();
                            let h_i = x_mesh[i+1]-x_mesh[i];
                            for k in 0..N {    
                            let x_new = x_mesh[i] + h_i*(k as f64 )/(N as f64);
                            new_grid.push(x_new);   }         
                        }
                        else {
                            let h_i = x_mesh[i+1]-x_mesh[i];
                            let x_new = x_mesh[i] + h_i/2.0;
                            new_grid.push(x_mesh[i]);
                            new_grid.push(x_new) ;
                        }    

            }  

   
log::info!("\n \n new_grid: {:?}",new_grid);  
log::info!("of length {}", new_grid.len());   
log::info!("\n \n new_initial_guess: {:?} of length {}", new_initial_guess, new_initial_guess.len());
log::info!("will be converted to DMatrix wirh number of columns {} and rows {}", new_grid.len(), n_rows);
assert_eq!(new_initial_guess.len(),new_grid.len()*n_rows);

let number_of_nonzero_keys = mark.iter().map(|(x, y)| if *y != 0 {1} else {0}).sum::<i32>() as usize;
log::info!("number of nonzero keys {}", number_of_nonzero_keys);
let  new_initial_guess:DMatrix<f64> =  DMatrix::from_vec(n_rows,new_grid.len(), new_initial_guess);

(new_grid, new_initial_guess,  number_of_nonzero_keys)

}

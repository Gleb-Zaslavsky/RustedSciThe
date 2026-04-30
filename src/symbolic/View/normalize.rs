//! Canonicalization and algebraic simplification for packed atoms.
//!
//! Normalization is the semantic backbone of the crate. Parsing and algebraic builders
//! are intentionally permissive and may create non-canonical packed atoms; normalization
//! sorts terms, merges coefficients, collapses powers, applies symmetry rules, and runs
//! lightweight builtin simplifications. The implementation keeps temporary storage small
//! with [`SmallVec`] and uses [`Workspace`] buffers to avoid repeated heap allocation.

use std::{cmp::Ordering, ops::DerefMut};

use smallvec::SmallVec;

use super::{
    atom::{Atom, AtomView, Fun, representation::InlineNum},
    coefficient::{Coefficient, CoefficientView},
    state::{RecycledAtom, State, Symbol, Workspace},
};

/// Normalize a borrowed atom view into `out`.
pub fn normalize_view(view: &AtomView<'_>, out: &mut Atom) {
    Workspace::get_local().with(|ws| view.normalize_with_ws(ws, out));
}

/// Return a normalized clone of an owned atom.
pub fn normalize(atom: &Atom) -> Atom {
    let mut result = Atom::new();
    normalize_view(&atom.as_view(), &mut result);
    result
}

impl<'a> AtomView<'a> {
    /// Compare two atom views in canonical normalization order.
    pub fn cmp(&self, other: &AtomView<'_>) -> Ordering {
        if self == other {
            return Ordering::Equal;
        }

        match (self, other) {
            (AtomView::Num(n1), AtomView::Num(n2)) => n1.get_coeff_view().cmp(&n2.get_coeff_view()),
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                let (b1, e1) = p1.get_base_exp();
                let (b2, e2) = p2.get_base_exp();
                b1.cmp(&b2).then_with(|| e1.cmp(&e2))
            }
            (_, AtomView::Pow(_)) => Ordering::Greater,
            (AtomView::Pow(_), _) => Ordering::Less,
            (AtomView::Mul(m1), AtomView::Mul(m2)) => {
                let it1 = m1.to_slice();
                let it2 = m2.to_slice();
                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }
                Ordering::Equal
            }
            (AtomView::Mul(_), _) => Ordering::Less,
            (_, AtomView::Mul(_)) => Ordering::Greater,
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();
                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }
                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,
            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_cmp = f1.get_symbol().cmp(&f2.get_symbol());
                if name_cmp != Ordering::Equal {
                    return name_cmp;
                }
                f1.fast_cmp(*f2)
            }
        }
    }

    pub(crate) fn cmp_factors(&self, other: &AtomView<'_>) -> Ordering {
        match (self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => p1.get_base().cmp(&p2.get_base()),
            (_, AtomView::Pow(p2)) => {
                let base = p2.get_base();
                self.cmp(&base).then(Ordering::Less)
            }
            (AtomView::Pow(p1), _) => {
                let base = p1.get_base();
                base.cmp(other).then(Ordering::Greater)
            }
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (AtomView::Mul(_), _) | (_, AtomView::Mul(_)) => {
                unreachable!("Cannot have submul in factor")
            }
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();
                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }
                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,
            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_cmp = f1.get_symbol().cmp(&f2.get_symbol());
                if name_cmp != Ordering::Equal {
                    return name_cmp;
                }
                f1.fast_cmp(*f2)
            }
        }
    }

    pub(crate) fn cmp_terms(&self, other: &AtomView<'_>) -> Ordering {
        debug_assert!(!matches!(self, AtomView::Add(_)));
        debug_assert!(!matches!(other, AtomView::Add(_)));
        match (self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                let (b1, e1) = p1.get_base_exp();
                let (b2, e2) = p2.get_base_exp();
                b1.cmp(&b2).then_with(|| e1.cmp(&e2))
            }
            (AtomView::Mul(m1), AtomView::Mul(m2)) => {
                let len1 = if m1.has_coefficient() {
                    m1.get_nargs() - 1
                } else {
                    m1.get_nargs()
                };
                let len2 = if m2.has_coefficient() {
                    m2.get_nargs() - 1
                } else {
                    m2.get_nargs()
                };
                let len_cmp = len1.cmp(&len2);
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (t1, t2) in m1.iter().zip(m2.iter()) {
                    if matches!(t1, AtomView::Num(_)) || matches!(t2, AtomView::Num(_)) {
                        break;
                    }
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }
                Ordering::Equal
            }
            (AtomView::Mul(m1), a2) => {
                if !m1.has_coefficient() || m1.get_nargs() != 2 {
                    return Ordering::Greater;
                }
                m1.to_slice().get(0).cmp(a2)
            }
            (a1, AtomView::Mul(m2)) => {
                if !m2.has_coefficient() || m2.get_nargs() != 2 {
                    return Ordering::Less;
                }
                a1.cmp(&m2.to_slice().get(0))
            }
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (_, AtomView::Pow(_)) => Ordering::Greater,
            (AtomView::Pow(_), _) => Ordering::Less,
            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_cmp = f1.get_symbol().cmp(&f2.get_symbol());
                if name_cmp != Ordering::Equal {
                    return name_cmp;
                }
                f1.fast_cmp(*f2)
            }
            (AtomView::Add(_), _) | (_, AtomView::Add(_)) => unreachable!("Cannot have nested add"),
        }
    }

    /// Return whether this atom is already marked as normalized.
    pub fn needs_normalization(&self) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => false,
            AtomView::Fun(f) => !f.is_normalized(),
            AtomView::Pow(p) => !p.is_normalized(),
            AtomView::Mul(m) => !m.is_normalized(),
            AtomView::Add(a) => !a.is_normalized(),
        }
    }

    /// Normalize this view using the thread-local workspace.
    pub fn normalize(&self, out: &mut Atom) {
        Workspace::get_local().with(|ws| self.normalize_with_ws(ws, out));
    }

    /// Normalize this view using an explicit workspace.
    pub(crate) fn normalize_with_ws(&self, workspace: &Workspace, out: &mut Atom) {
        if !self.needs_normalization() {
            out.set_from_view(self);
            return;
        }

        match self {
            AtomView::Mul(t) => {
                let mut atom_buf: SmallVec<[_; 20]> = SmallVec::new();
                let mut is_zero = false;

                for a in t.iter() {
                    let mut handle = workspace.new_atom();
                    if a.needs_normalization() {
                        a.normalize_with_ws(workspace, &mut handle);
                    } else {
                        handle.set_from_view(&a);
                    }

                    if let Atom::Mul(mul) = handle.deref_mut() {
                        for c in mul.to_mul_view().iter() {
                            let mut inner = workspace.new_atom();
                            inner.set_from_view(&c);
                            if let AtomView::Num(n) = c {
                                if n.is_one() {
                                    continue;
                                }
                                if n.is_zero() {
                                    out.to_num(Coefficient::zero());
                                    is_zero = true;
                                    continue;
                                }
                            }
                            atom_buf.push(inner);
                        }
                    } else {
                        if let AtomView::Num(n) = handle.as_view() {
                            if n.is_one() {
                                continue;
                            }
                            if n.is_zero() {
                                out.to_num(Coefficient::zero());
                                is_zero = true;
                                continue;
                            }
                        }
                        atom_buf.push(handle);
                    }
                }

                if is_zero {
                    return;
                }

                atom_buf.sort_by(|a, b| a.as_view().cmp_factors(&b.as_view()));

                if atom_buf.is_empty() {
                    out.to_num(1.into());
                    return;
                }

                let out_mul = out.to_mul();
                atom_buf.reverse();
                let mut last_buf = atom_buf.pop().unwrap();
                let mut tmp = workspace.new_atom();
                let mut cur_len = 0;

                while let Some(mut cur_buf) = atom_buf.pop() {
                    if !last_buf.merge_factors(&mut cur_buf, &mut tmp, workspace) {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_one() {
                                atom_buf.insert(0, last_buf);
                            }
                        } else {
                            out_mul.extend(v);
                            cur_len += 1;
                        }
                        last_buf = cur_buf;
                    }
                }

                if cur_len == 0 {
                    out.set_from_view(&last_buf.as_view());
                } else {
                    let v = last_buf.as_view();
                    if let AtomView::Num(n) = v {
                        if !n.is_one() {
                            out_mul.extend(v);
                            out_mul.set_has_coefficient(true);
                            out_mul.set_normalized(true);
                        } else if cur_len == 1 {
                            last_buf.set_from_view(&out_mul.to_mul_view().to_slice().get(0));
                            out.set_from_view(&last_buf.as_view());
                        } else {
                            out_mul.set_normalized(true);
                        }
                    } else {
                        out_mul.extend(v);
                        out_mul.set_normalized(true);
                    }
                }
            }
            AtomView::Num(n) => {
                out.to_num(n.get_coeff_view().normalize());
            }
            AtomView::Var(_) => self.clone_into(out),
            AtomView::Fun(f) => {
                let id = f.get_symbol();
                let out_f = out.to_fun(id);

                fn add_arg(f: &mut Fun, a: AtomView<'_>) {
                    if let AtomView::Fun(fa) = a {
                        if fa.get_symbol() == Atom::ARG {
                            for aa in fa.iter() {
                                f.add_arg(aa);
                            }
                            return;
                        }
                    }
                    f.add_arg(a);
                }

                fn cartesian_product<'b>(
                    workspace: &Workspace,
                    list: &[Vec<AtomView<'b>>],
                    fun_name: Symbol,
                    cur: &mut Vec<AtomView<'b>>,
                    acc: &mut Vec<RecycledAtom>,
                ) {
                    if list.is_empty() {
                        let mut h = workspace.new_atom();
                        let f = h.to_fun(fun_name);
                        for a in cur.iter() {
                            add_arg(f, *a);
                        }
                        acc.push(h);
                        return;
                    }

                    for a in &list[0] {
                        cur.push(*a);
                        cartesian_product(workspace, &list[1..], fun_name, cur, acc);
                        cur.pop();
                    }
                }

                let mut handle = workspace.new_atom();
                for a in f {
                    if a.needs_normalization() {
                        a.normalize_with_ws(workspace, &mut handle);
                        add_arg(out_f, handle.as_view());
                    } else {
                        add_arg(out_f, a);
                    }
                }
                out_f.set_normalized(true);

                if [
                    Atom::COS,
                    Atom::SIN,
                    Atom::TAN,
                    Atom::ASIN,
                    Atom::ATAN,
                    Atom::EXP,
                    Atom::LOG,
                    Atom::ACOS,
                ]
                .contains(&id)
                    && out_f.to_fun_view().get_nargs() == 1
                {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if let AtomView::Num(n) = arg {
                        if n.is_zero() {
                            if id == Atom::COS || id == Atom::EXP {
                                out.to_num(Coefficient::one());
                                return;
                            } else if [Atom::SIN, Atom::TAN, Atom::ASIN, Atom::ATAN].contains(&id) {
                                out.to_num(Coefficient::zero());
                                return;
                            }
                        }
                        if n.is_one() && (id == Atom::LOG || id == Atom::ACOS) {
                            out.to_num(Coefficient::zero());
                            return;
                        }
                    }
                }

                if id == Atom::EXP && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if contains_symbol(arg, Atom::LOG) {
                        let mut buffer = workspace.new_atom();
                        if arg.simplify_exp_log(workspace, &mut buffer) {
                            out.set_from_view(&buffer.as_view());
                            return;
                        }
                    }
                }

                if id.is_linear() {
                    if out_f
                        .to_fun_view()
                        .iter()
                        .any(|a| matches!(a, AtomView::Add(_)))
                    {
                        let mut arg_buf = Vec::with_capacity(out_f.to_fun_view().get_nargs());
                        for a in out_f.to_fun_view().iter() {
                            let mut vec = vec![];
                            if let AtomView::Add(aa) = a {
                                for x in aa.iter() {
                                    vec.push(x);
                                }
                            } else {
                                vec.push(a);
                            }
                            arg_buf.push(vec);
                        }

                        let mut acc = Vec::new();
                        cartesian_product(workspace, &arg_buf, id, &mut vec![], &mut acc);
                        let mut add_h = workspace.new_atom();
                        let add = add_h.to_add();
                        let mut h = workspace.new_atom();
                        for a in acc {
                            a.as_view().normalize_with_ws(workspace, &mut h);
                            add.extend(h.as_view());
                        }
                        add_h.as_view().normalize_with_ws(workspace, out);
                        return;
                    }

                    if out_f
                        .to_fun_view()
                        .iter()
                        .any(|a| matches!(a, AtomView::Mul(m) if m.has_coefficient()))
                    {
                        let mut new_term = workspace.new_atom();
                        let t = new_term.to_mul();
                        let mut new_fun = workspace.new_atom();
                        let nf = new_fun.to_fun(id);
                        let mut coeff: Coefficient = 1.into();
                        for a in out_f.to_fun_view().iter() {
                            if let AtomView::Mul(m) = a {
                                if m.has_coefficient() {
                                    let mut stripped = workspace.new_atom();
                                    let mul = stripped.to_mul();
                                    for x in m {
                                        if let AtomView::Num(n) = x {
                                            coeff = coeff * n.get_coeff_view().to_owned();
                                        } else {
                                            mul.extend(x);
                                        }
                                    }
                                    nf.add_arg(stripped.as_view());
                                } else {
                                    nf.add_arg(a);
                                }
                            } else {
                                nf.add_arg(a);
                            }
                        }
                        t.extend(new_fun.as_view());
                        t.extend(workspace.new_num(coeff).as_view());
                        t.as_view().normalize_with_ws(workspace, out);
                        return;
                    }

                    for a in out_f.to_fun_view() {
                        if let AtomView::Num(n) = a {
                            if n.is_zero() {
                                out.to_num(Coefficient::zero());
                                return;
                            }
                        }
                    }
                }

                if id.is_symmetric() || id.is_antisymmetric() {
                    let mut arg_buf: SmallVec<[(usize, _); 20]> = SmallVec::new();
                    for (i, a) in out_f.to_fun_view().iter().enumerate() {
                        let mut handle = workspace.new_atom();
                        handle.set_from_view(&a);
                        arg_buf.push((i, handle));
                    }
                    arg_buf.sort_by(|a, b| a.1.as_view().cmp(&b.1.as_view()));

                    if id.is_antisymmetric() {
                        if arg_buf
                            .windows(2)
                            .any(|w| w[0].1.as_view() == w[1].1.as_view())
                        {
                            out.to_num(Coefficient::zero());
                            return;
                        }

                        let mut order: SmallVec<[usize; 20]> = (0..arg_buf.len())
                            .map(|i| arg_buf.iter().position(|(j, _)| *j == i).unwrap())
                            .collect();
                        let mut swaps = 0;
                        for i in 0..order.len() {
                            let pos = order[i..].iter().position(|&x| x == i).unwrap();
                            order.copy_within(i..i + pos, i + 1);
                            swaps += pos;
                        }

                        if swaps % 2 == 1 {
                            let mut handle = workspace.new_atom();
                            let out_f = handle.to_fun(id);
                            for (_, a) in arg_buf {
                                out_f.add_arg(a.as_view());
                            }
                            out_f.set_normalized(true);

                            if let Some(f) = State::get_normalization_function(id) {
                                let mut fs = workspace.new_atom();
                                if f(handle.as_view(), &mut fs) {
                                    std::mem::swap(&mut handle, &mut fs);
                                }
                            }

                            let m = out.to_mul();
                            m.extend(handle.as_view());
                            handle.to_num((-1).into());
                            m.extend(handle.as_view());
                            m.set_normalized(true);
                            return;
                        }
                    }

                    let out_f = out.to_fun(id);
                    for (_, a) in arg_buf {
                        out_f.add_arg(a.as_view());
                    }
                    out_f.set_normalized(true);
                } else if id.is_cyclesymmetric() {
                    let mut args: SmallVec<[_; 20]> = SmallVec::new();
                    for a in out_f.to_fun_view().iter() {
                        args.push(a);
                    }

                    let mut best_shift = 0;
                    'shift: for shift in 1..args.len() {
                        for i in 0..args.len() {
                            match args[(i + best_shift) % args.len()]
                                .cmp(&args[(i + shift) % args.len()])
                            {
                                Ordering::Equal => {}
                                Ordering::Less => continue 'shift,
                                Ordering::Greater => break,
                            }
                        }
                        best_shift = shift;
                    }

                    let mut f2 = workspace.new_atom();
                    let ff = f2.to_fun(id);
                    for arg in args[best_shift..].iter().chain(&args[..best_shift]) {
                        ff.add_arg(*arg);
                    }
                    ff.set_normalized(true);
                    drop(args);
                    std::mem::swap(ff, out_f);
                }

                if let Some(f) = State::get_normalization_function(id) {
                    let mut fs = workspace.new_atom();
                    if f(out.as_view(), &mut fs) {
                        std::mem::swap(out, fs.deref_mut());
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let mut base_handle = workspace.new_atom();
                let mut exp_handle = workspace.new_atom();

                if base.needs_normalization() {
                    base.normalize_with_ws(workspace, &mut base_handle);
                } else {
                    base_handle.set_from_view(&base);
                }
                if exp.needs_normalization() {
                    exp.normalize_with_ws(workspace, &mut exp_handle);
                } else {
                    exp_handle.set_from_view(&exp);
                }

                'pow_simplify: {
                    if base_handle.is_one() {
                        out.to_num(1.into());
                        break 'pow_simplify;
                    }

                    if let AtomView::Num(e) = exp_handle.as_view() {
                        let exp_num = e.get_coeff_view();
                        if exp_num == CoefficientView::Natural(0, 1) {
                            out.to_num(1.into());
                            break 'pow_simplify;
                        } else if exp_num == CoefficientView::Natural(1, 1) {
                            out.set_from_view(&base_handle.as_view());
                            break 'pow_simplify;
                        } else if let AtomView::Num(n) = base_handle.as_view() {
                            let (new_base_num, new_exp_num) = n.get_coeff_view().pow(&exp_num);
                            if new_exp_num == 1.into() {
                                out.to_num(new_base_num);
                                break 'pow_simplify;
                            }
                            base_handle.to_num(new_base_num);
                            exp_handle.to_num(new_exp_num);
                        } else if let AtomView::Var(v) = base_handle.as_view() {
                            if v.get_symbol() == Atom::I {
                                if let CoefficientView::Natural(n, d) = exp_num {
                                    let mut new_base = workspace.new_atom();
                                    if n % 2 == 0 {
                                        if n % 4 == 0 {
                                            new_base.to_num(1.into());
                                        } else {
                                            new_base.to_num((-1).into());
                                        }
                                    } else if (n - 1) % 4 == 0 {
                                        new_base.set_from_view(&base_handle.as_view());
                                    } else {
                                        let m = new_base.to_mul();
                                        m.extend(base_handle.as_view());
                                        let mut helper = workspace.new_atom();
                                        helper.to_num((-1).into());
                                        m.extend(helper.as_view());
                                        new_base
                                            .as_view()
                                            .normalize_with_ws(workspace, &mut helper);
                                        std::mem::swap(&mut new_base, &mut helper);
                                    }

                                    if d == 1 {
                                        out.set_from_view(&new_base.as_view());
                                    } else {
                                        let mut new_exp = workspace.new_atom();
                                        new_exp.to_num((1i64, d).into());
                                        out.to_pow(new_base.as_view(), new_exp.as_view());
                                    }
                                    break 'pow_simplify;
                                }
                            }
                        } else if let AtomView::Pow(p_base) = base_handle.as_view() {
                            if exp_num.is_integer() {
                                let (p_base_base, p_base_exp) = p_base.get_base_exp();
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(p_base_exp);
                                mul.extend(exp_handle.as_view());
                                let mut exp_h = workspace.new_atom();
                                mul.as_view().normalize_with_ws(workspace, &mut exp_h);
                                mul_h.to_pow(p_base_base, exp_h.as_view());
                                mul_h.as_view().normalize_with_ws(workspace, out);
                                break 'pow_simplify;
                            }
                        } else if let AtomView::Mul(m) = base_handle.as_view() {
                            if exp_num.is_integer() {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                for arg in m {
                                    let mut pow_h = workspace.new_atom();
                                    pow_h.to_pow(arg, exp_handle.as_view());
                                    mul.extend(pow_h.as_view());
                                }
                                mul_h.as_view().normalize_with_ws(workspace, out);
                                break 'pow_simplify;
                            }
                        }
                    }
                    out.to_pow(base_handle.as_view(), exp_handle.as_view());
                }
                out.set_normalized(true);
            }
            AtomView::Add(a) => {
                let mut new_sum = workspace.new_atom();
                let ns = new_sum.to_add();
                let mut atom_sort_buf: SmallVec<[_; 20]> = SmallVec::new();
                let mut norm_arg = workspace.new_atom();

                for term in a {
                    let r = if term.needs_normalization() {
                        term.normalize_with_ws(workspace, &mut norm_arg);
                        norm_arg.as_view()
                    } else {
                        term
                    };

                    if let AtomView::Add(new_add) = r {
                        for c in new_add.iter() {
                            if let AtomView::Num(n) = c {
                                if n.is_zero() {
                                    continue;
                                }
                            }
                            ns.extend(c);
                        }
                    } else {
                        if let AtomView::Num(n) = r {
                            if n.is_zero() {
                                continue;
                            }
                        }
                        ns.extend(r);
                    }
                }

                for x in ns.to_add_view().iter() {
                    atom_sort_buf.push(x);
                }
                atom_sort_buf.sort_by(|a, b| a.cmp_terms(b));

                if atom_sort_buf.is_empty() {
                    out.to_num(Coefficient::zero());
                    return;
                }

                let out_add = out.to_add();
                let mut last_buf = workspace.new_atom();
                last_buf.set_from_view(&atom_sort_buf[0]);
                let mut helper = workspace.new_atom();
                let mut cur_len = 0;

                for cur in atom_sort_buf.iter().skip(1) {
                    if !last_buf.merge_terms(*cur, &mut helper) {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                out_add.extend(v);
                                cur_len += 1;
                            }
                        } else {
                            out_add.extend(v);
                            cur_len += 1;
                        }
                        cur.clone_into(&mut last_buf);
                    }
                }

                if cur_len == 0 {
                    out.set_from_view(&last_buf.as_view());
                } else {
                    let v = last_buf.as_view();
                    if let AtomView::Num(n) = v {
                        if !n.is_zero() {
                            out_add.extend(v);
                            out_add.set_normalized(true);
                        } else if cur_len == 1 {
                            last_buf.set_from_view(&out_add.to_add_view().to_slice().get(0));
                            out.set_from_view(&last_buf.as_view());
                        } else {
                            out_add.set_normalized(true);
                        }
                    } else {
                        out_add.extend(v);
                        out_add.set_normalized(true);
                    }
                }
            }
        }
    }

    fn simplify_exp_log(&self, ws: &Workspace, out: &mut Atom) -> bool {
        if let AtomView::Fun(f) = self {
            if f.get_symbol() == Atom::LOG && f.get_nargs() == 1 {
                out.set_from_view(&f.iter().next().unwrap());
                return true;
            }
        }

        if let AtomView::Mul(m) = self {
            let mut found_index = None;
            for (i, a) in m.iter().enumerate() {
                if a.simplify_exp_log(ws, out) {
                    if found_index.is_some() {
                        return false;
                    }
                    found_index = Some(i);
                }
            }

            if let Some(i) = found_index {
                let mut mm = ws.new_atom();
                let mmm = mm.to_mul();
                for (j, a) in m.iter().enumerate() {
                    if j != i {
                        mmm.extend(a);
                    }
                }
                let mut b = ws.new_atom();
                b.to_pow(out.as_view(), mm.as_view());
                b.as_view().normalize_with_ws(ws, out);
                return true;
            }
            return false;
        }

        if let AtomView::Add(a) = self {
            let mut mm = ws.new_atom();
            let m = mm.to_mul();
            let mut aa = ws.new_atom();
            let aaa = aa.to_add();
            let mut changed = false;
            for term in a {
                if term.simplify_exp_log(ws, out) {
                    changed = true;
                    m.extend(out.as_view());
                } else {
                    aaa.extend(term);
                }
            }
            if changed {
                let mut new_exp = ws.new_atom();
                new_exp.to_fun(Atom::EXP).add_arg(aa.as_view());
                m.extend(new_exp.as_view());
                mm.as_view().normalize_with_ws(ws, out);
            }
            return changed;
        }

        false
    }
}

impl Atom {
    fn merge_factors(
        &mut self,
        other: &mut Self,
        helper: &mut Self,
        workspace: &Workspace,
    ) -> bool {
        if let Atom::Pow(p1) = self {
            if let Atom::Pow(p2) = other {
                let (base2, exp2) = p2.to_pow_view().get_base_exp();
                let (base1, exp1) = p1.to_pow_view().get_base_exp();
                if base1 != base2 {
                    return false;
                }
                if let AtomView::Num(n1) = exp1 {
                    if let AtomView::Num(n2) = exp2 {
                        let new_exp = helper.to_num(n1.get_coeff_view() + n2.get_coeff_view());
                        if new_exp.to_num_view().is_zero() {
                            self.to_num(1.into());
                        } else if new_exp.to_num_view().is_one() {
                            self.set_from_view(&base2);
                        } else {
                            p1.set_from_base_and_exp(base2, helper.as_view());
                        }
                        self.set_normalized(true);
                        return true;
                    }
                }

                let new_exp = helper.to_add();
                new_exp.extend(exp1);
                new_exp.extend(exp2);
                let mut helper2 = workspace.new_atom();
                helper.as_view().normalize_with_ws(workspace, &mut helper2);
                if let AtomView::Num(n) = helper2.as_view() {
                    if n.is_zero() {
                        self.to_num(1.into());
                        return true;
                    } else if n.is_one() {
                        self.set_from_view(&base2);
                        return true;
                    }
                }
                p1.set_from_base_and_exp(base2, helper2.as_view());
                p1.set_normalized(true);
                return true;
            }

            let (base, exp) = p1.to_pow_view().get_base_exp();
            if other.as_view() == base {
                if let AtomView::Num(n) = exp {
                    let new_exp = n.get_coeff_view() + 1;
                    if new_exp.is_zero() {
                        self.to_num(1.into());
                    } else if new_exp == 1.into() {
                        self.set_from_view(&other.as_view());
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(other.as_view(), AtomView::Num(num.to_num_view()));
                    }
                } else {
                    other.to_num(1.into());
                    let new_exp = helper.to_add();
                    new_exp.extend(other.as_view());
                    new_exp.extend(exp);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize_with_ws(workspace, &mut helper2);
                    other.to_pow(base, helper2.as_view());
                    std::mem::swap(self, other);
                }
                self.set_normalized(true);
                return true;
            }
            return false;
        }

        if let Atom::Pow(p) = other {
            let (base, exp) = p.to_pow_view().get_base_exp();
            if self.as_view() == base {
                if let AtomView::Num(n) = exp {
                    let new_exp = n.get_coeff_view() + 1;
                    if new_exp.is_zero() {
                        self.to_num(1.into());
                    } else if new_exp == 1.into() {
                        self.set_from_view(&base);
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(base, AtomView::Num(num.to_num_view()));
                    }
                } else {
                    self.to_num(1.into());
                    let new_exp = helper.to_add();
                    new_exp.extend(self.as_view());
                    new_exp.extend(exp);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize_with_ws(workspace, &mut helper2);
                    self.to_pow(base, helper2.as_view());
                }
                self.set_normalized(true);
                return true;
            }
            return false;
        }

        if let Atom::Num(n1) = self {
            if let Atom::Num(n2) = other {
                n1.mul(&n2.to_num_view());
                return true;
            }
            return false;
        }

        if self.as_view() == other.as_view() {
            if let AtomView::Var(v) = self.as_view() {
                if v.get_symbol() == Atom::I {
                    self.to_num((-1).into());
                    return true;
                }
            }
            let exp = other.to_num(2.into());
            helper.to_pow(self.as_view(), AtomView::Num(exp.to_num_view()));
            helper.set_normalized(true);
            std::mem::swap(self, helper);
            return true;
        }

        false
    }

    pub(crate) fn merge_terms(&mut self, other: AtomView<'_>, helper: &mut Self) -> bool {
        if let Atom::Num(n1) = self {
            if let AtomView::Num(n2) = other {
                n1.add(&n2);
                return true;
            }
            return false;
        }

        if let Atom::Mul(m) = self {
            let slice = m.to_mul_view().to_slice();
            let last_elem = slice.get(slice.len() - 1);
            let (non_coeff1, has_coeff) = if matches!(last_elem, AtomView::Num(_)) {
                (slice.get_subslice(0..slice.len() - 1), true)
            } else {
                (m.to_mul_view().to_slice(), false)
            };

            if let AtomView::Mul(m2) = other {
                let slice2 = m2.to_slice();
                let last_elem2 = slice2.get(slice2.len() - 1);
                let non_coeff2 = if matches!(last_elem2, AtomView::Num(_)) {
                    slice2.get_subslice(0..slice2.len() - 1)
                } else {
                    m2.to_slice()
                };

                if non_coeff1.eq(&non_coeff2) {
                    let num = if let AtomView::Num(n) = last_elem {
                        n.get_coeff_view()
                    } else {
                        CoefficientView::Natural(1, 1)
                    };
                    let new_coeff = if let AtomView::Num(n) = last_elem2 {
                        num + n.get_coeff_view()
                    } else {
                        num + 1
                    };
                    let len = slice.len();

                    if new_coeff == 1.into() {
                        if has_coeff {
                            if len == 2 {
                                self.set_from_view(&slice2.get(0));
                            } else {
                                let m = self.to_mul();
                                for a in non_coeff2.iter() {
                                    m.extend(a);
                                }
                                m.set_has_coefficient(false);
                                m.set_normalized(true);
                            }
                        }
                        return true;
                    }

                    if new_coeff.is_zero() {
                        self.to_num(new_coeff);
                        return true;
                    }

                    let on = helper.to_num(new_coeff);
                    if has_coeff {
                        m.replace_last(on.to_num_view().as_view());
                    } else {
                        m.extend(on.to_num_view().as_view());
                    }
                    m.set_has_coefficient(true);
                    m.set_normalized(true);
                    return true;
                }
            } else {
                if non_coeff1.len() != 1 || other != slice.get(0) {
                    return false;
                }
                let new_coeff = if let AtomView::Num(n) = last_elem {
                    n.get_coeff_view() + 1
                } else {
                    return false;
                };
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }
                let on = helper.to_num(new_coeff);
                m.replace_last(on.to_num_view().as_view());
                m.set_normalized(true);
                return true;
            }
        } else if let AtomView::Mul(m) = other {
            let slice = m.to_slice();
            if slice.len() != 2 {
                return false;
            }
            let last_elem = slice.get(slice.len() - 1);
            if self.as_view() == slice.get(0) {
                let (new_coeff, has_num) = if let AtomView::Num(n) = last_elem {
                    (n.get_coeff_view() + 1, true)
                } else {
                    return false;
                };
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }
                let on = helper.to_num(new_coeff);
                other.clone_into(self);
                if let Atom::Mul(m) = self {
                    if has_num {
                        m.replace_last(on.to_num_view().as_view());
                    } else {
                        m.extend(on.to_num_view().as_view());
                    }
                    m.set_has_coefficient(true);
                }
                self.set_normalized(true);
                return true;
            }
        } else if self.as_view() == other {
            let mul = helper.to_mul();
            mul.extend(self.as_view());
            self.to_num((2, 1).into());
            mul.extend(self.as_view());
            mul.set_has_coefficient(true);
            mul.set_normalized(true);
            std::mem::swap(self, helper);
            return true;
        }

        false
    }
}

impl<'a> AtomView<'a> {
    #[allow(dead_code)]
    pub(crate) fn add_normalized(&self, rhs: AtomView<'_>, ws: &Workspace, out: &mut Atom) {
        if *self == rhs {
            let mut a = ws.new_atom();
            let m = a.to_mul();
            m.extend(*self);
            m.extend(InlineNum::new(2, 1).as_view());
            a.as_view().normalize_with_ws(ws, out);
            return;
        }

        let a = out.to_add();
        a.grow_capacity(self.get_byte_size() + rhs.get_byte_size());

        let mut helper = ws.new_atom();
        let mut b = ws.new_atom();
        if let AtomView::Add(a1) = self {
            if let AtomView::Add(a2) = rhs {
                let mut s = a1.iter();
                let mut t = a2.iter();
                let mut curs = s.next();
                let mut curst = t.next();
                while curs.is_some() || curst.is_some() {
                    if let Some(ss) = curs {
                        if let Some(tt) = curst {
                            match ss.cmp_terms(&tt) {
                                Ordering::Less => {
                                    a.extend(ss);
                                    curs = s.next();
                                }
                                Ordering::Greater => {
                                    a.extend(tt);
                                    curst = t.next();
                                }
                                Ordering::Equal => {
                                    b.set_from_view(&ss);
                                    if b.merge_terms(tt, &mut helper) {
                                        if let AtomView::Num(n) = b.as_view() {
                                            if !n.is_zero() {
                                                a.extend(b.as_view());
                                            }
                                        } else {
                                            a.extend(b.as_view());
                                        }
                                    }
                                    curst = t.next();
                                    curs = s.next();
                                }
                            }
                        } else {
                            a.extend(ss);
                            curs = s.next();
                        }
                    } else if let Some(tt) = curst {
                        a.extend(tt);
                        curst = t.next();
                    }
                }

                if a.get_nargs() == 0 {
                    out.to_num(Coefficient::zero());
                } else if a.get_nargs() == 1 {
                    let mut b2 = ws.new_atom();
                    b2.set_from_view(&a.to_add_view().iter().next().unwrap());
                    out.set_from_view(&b2.as_view());
                } else {
                    a.set_normalized(true);
                }
                return;
            }
        }

        if let AtomView::Add(a1) = self {
            if rhs.is_zero() {
                self.clone_into(out);
                return;
            }

            if a1.get_nargs() < 50 {
                let mut found = false;
                for x in a1.iter() {
                    if found {
                        a.extend(x);
                        continue;
                    }
                    match x.cmp_terms(&rhs) {
                        Ordering::Less => a.extend(x),
                        Ordering::Equal => {
                            found = true;
                            b.set_from_view(&x);
                            if b.merge_terms(rhs, &mut helper) {
                                if let AtomView::Num(n) = b.as_view() {
                                    if !n.is_zero() {
                                        a.extend(b.as_view());
                                    }
                                } else {
                                    a.extend(b.as_view());
                                }
                            }
                        }
                        Ordering::Greater => {
                            found = true;
                            a.extend(rhs);
                            a.extend(x);
                        }
                    }
                }
                if !found {
                    a.extend(rhs);
                }
            } else {
                let v: Vec<_> = a1.iter().collect();
                match v.binary_search_by(|a| a.cmp_terms(&rhs)) {
                    Ok(p) => {
                        for x in v.iter().take(p) {
                            a.extend(*x);
                        }
                        b.set_from_view(&v[p]);
                        if b.merge_terms(rhs, &mut helper) {
                            if let AtomView::Num(n) = b.as_view() {
                                if !n.is_zero() {
                                    a.extend(b.as_view());
                                }
                            } else {
                                a.extend(b.as_view());
                            }
                        }
                        for x in v.iter().skip(p + 1) {
                            a.extend(*x);
                        }
                    }
                    Err(p) => {
                        for x in v.iter().take(p) {
                            a.extend(*x);
                        }
                        a.extend(rhs);
                        for x in v.iter().skip(p) {
                            a.extend(*x);
                        }
                    }
                }
            }

            if a.get_nargs() == 1 {
                let mut b2 = ws.new_atom();
                b2.set_from_view(&a.to_add_view().iter().next().unwrap());
                out.set_from_view(&b2.as_view());
            } else {
                a.set_normalized(true);
            }
        } else if let AtomView::Add(_) = rhs {
            rhs.add_normalized(*self, ws, out);
        } else {
            let mut e = ws.new_atom();
            let a2 = e.to_add();
            a2.extend(*self);
            a2.extend(rhs);
            e.as_view().normalize_with_ws(ws, out);
        }
    }
}

fn contains_symbol(view: AtomView<'_>, symbol: Symbol) -> bool {
    match view {
        AtomView::Num(_) => false,
        AtomView::Var(v) => v.get_symbol() == symbol,
        AtomView::Fun(f) => {
            f.get_symbol() == symbol || f.iter().any(|a| contains_symbol(a, symbol))
        }
        AtomView::Pow(p) => {
            let (b, e) = p.get_base_exp();
            contains_symbol(b, symbol) || contains_symbol(e, symbol)
        }
        AtomView::Mul(m) => m.iter().any(|a| contains_symbol(a, symbol)),
        AtomView::Add(a) => a.iter().any(|x| contains_symbol(x, symbol)),
    }
}

#[cfg(test)]
mod test {
    use crate::symbolic::View::atom::{Atom, FunctionBuilder};
    use crate::{function, parse, symbol};
    #[test]
    fn pow_apart() {
        let res = parse!("v1*(v1*v2*v3)^-5").unwrap();
        let refr = parse!("v1^-4*v2^-5*v3^-5").unwrap();
        assert_eq!(res, refr);
    }

    #[test]
    fn pow_simplify() {
        assert_eq!(parse!("1^(1/2)").unwrap(), parse!("1").unwrap());
        assert_eq!(parse!("(v1^v2)^2").unwrap(), parse!("v1^(2*v2)").unwrap());
        assert_eq!(parse!("(v1^(1/2))^2").unwrap(), parse!("v1").unwrap());
    }

    #[test]
    fn linear_symmetric() {
        let f = crate::symbolic::View::state::Symbol::new_with_attributes(
            crate::wrap_symbol!("fsl1"),
            &[
                crate::symbolic::View::atom::FunctionAttribute::Symmetric,
                crate::symbolic::View::atom::FunctionAttribute::Linear,
            ],
        )
        .unwrap();
        let res = parse!("fsl1(v2+2*v3,v1+3*v2-v3)").unwrap();
        let refr = function!(f, symbol!("v1"), symbol!("v2"))
            + function!(f, symbol!("v1"), symbol!("v3")) * 2
            + function!(f, symbol!("v2"), symbol!("v2")) * 3
            + function!(f, symbol!("v2"), symbol!("v3")) * 5
            - function!(f, symbol!("v3"), symbol!("v3")) * 2;
        assert_eq!(res, refr);
    }

    #[test]
    fn mul_complex_i() {
        let res = Atom::new_var(Atom::I) * Atom::new_var(Atom::E) * Atom::new_var(Atom::I);
        let refr = -Atom::new_var(Atom::E);
        assert_eq!(res, refr);
    }

    #[test]
    fn exp_log_simplify() {
        assert_eq!(parse!("exp(log(v1))").unwrap(), parse!("v1").unwrap());
    }
    #[test]
    fn extended_builtin_constant_simplifications() {
        assert_eq!(parse!("tan(0)").unwrap(), parse!("0").unwrap());
        assert_eq!(parse!("asin(0)").unwrap(), parse!("0").unwrap());
        assert_eq!(parse!("atan(0)").unwrap(), parse!("0").unwrap());
        assert_eq!(parse!("acos(1)").unwrap(), parse!("0").unwrap());
    }

    #[test]
    fn symmetric_function_reorders_arguments() {
        let f = crate::symbolic::View::state::Symbol::new_with_attributes(
            crate::wrap_symbol!("fsym"),
            &[crate::symbolic::View::atom::FunctionAttribute::Symmetric],
        )
        .unwrap();
        let a = function!(f, symbol!("v2"), symbol!("v1"), symbol!("v3"));
        let b = function!(f, symbol!("v1"), symbol!("v2"), symbol!("v3"));
        assert_eq!(a, b);
    }

    #[test]
    fn cyclesymmetric_function_rotates_to_canonical_form() {
        let f = crate::symbolic::View::state::Symbol::new_with_attributes(
            crate::wrap_symbol!("fcyc"),
            &[crate::symbolic::View::atom::FunctionAttribute::Cyclesymmetric],
        )
        .unwrap();
        let a = function!(f, symbol!("v2"), symbol!("v3"), symbol!("v1"));
        let b = function!(f, symbol!("v1"), symbol!("v2"), symbol!("v3"));
        assert_eq!(a, b);
    }

    #[test]
    fn coeff_flag() {
        let a = parse!("(-v1)*v2").unwrap();
        if let Atom::Mul(m) = &a {
            assert!(m.to_mul_view().has_coefficient());
            assert!(m.to_mul_view().is_normalized());
        } else {
            panic!("Expected Mul");
        }

        let b = a * parse!("-v3").unwrap();
        if let Atom::Mul(m) = &b {
            assert!(!m.to_mul_view().has_coefficient());
            assert!(m.to_mul_view().is_normalized());
        } else {
            panic!("Expected Mul");
        }
    }

    #[test]
    fn add_normalized() {
        let a = parse!("v1+v2+v3").unwrap();
        let b = parse!("1+v2+v4+v5").unwrap();
        assert_eq!(a + b, parse!("v1+2*v2+v3+v4+v5+1").unwrap());
    }

    #[test]
    fn builder_style_expression_normalizes() {
        let x = symbol!("x");
        let y = symbol!("y");
        let f = symbol!("f");
        let fun = FunctionBuilder::new(f)
            .add_arg(Atom::new_var(x))
            .add_arg(Atom::new_var(y))
            .add_arg(Atom::new_num(2))
            .finish();
        let expr = (-(Atom::new_var(y) + Atom::new_var(x) + 2) * Atom::new_var(y) * 6).npow(2)
            / Atom::new_var(y)
            * fun
            / 4;
        let expected = parse!("1/4*y^-1*(-6*(x+y+2)*y)^2*f(x,y,2)").unwrap();
        assert_eq!(expr, expected);
    }
}

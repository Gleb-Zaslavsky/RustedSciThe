use super::error::BandedError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMajorLayout {
    n_nodes: usize,
    vars_per_node: usize,
}

impl NodeMajorLayout {
    pub fn new(n_nodes: usize, vars_per_node: usize) -> Result<Self, BandedError> {
        if n_nodes == 0 || vars_per_node == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        Ok(Self {
            n_nodes,
            vars_per_node,
        })
    }

    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    pub fn vars_per_node(&self) -> usize {
        self.vars_per_node
    }

    #[inline]
    pub fn block_size(&self) -> usize {
        self.vars_per_node
    }

    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_nodes * self.vars_per_node
    }

    #[inline]
    pub fn global_index(&self, node: usize, local_var: usize) -> Result<usize, BandedError> {
        if node >= self.n_nodes || local_var >= self.vars_per_node {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(node * self.vars_per_node + local_var)
    }

    #[inline]
    pub fn node_of(&self, global: usize) -> Result<usize, BandedError> {
        if global >= self.n_total() {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(global / self.vars_per_node)
    }

    #[inline]
    pub fn local_var_of(&self, global: usize) -> Result<usize, BandedError> {
        if global >= self.n_total() {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(global % self.vars_per_node)
    }

    #[inline]
    pub fn split_global(&self, global: usize) -> Result<(usize, usize), BandedError> {
        if global >= self.n_total() {
            return Err(BandedError::DimensionMismatch);
        }
        Ok((global / self.vars_per_node, global % self.vars_per_node))
    }

    #[inline]
    pub fn block_row_of(&self, global_row: usize) -> Result<usize, BandedError> {
        self.node_of(global_row)
    }

    #[inline]
    pub fn block_col_of(&self, global_col: usize) -> Result<usize, BandedError> {
        self.node_of(global_col)
    }

    #[inline]
    pub fn same_block(&self, i: usize, j: usize) -> Result<bool, BandedError> {
        Ok(self.block_row_of(i)? == self.block_col_of(j)?)
    }

    /// Returns the block-distance between two global indices:
    ///   block(node_of(j)) - block(node_of(i))
    ///
    /// 0 means same node block, ±1 means neighboring node block, etc.
    #[inline]
    pub fn block_offset_of(&self, i: usize, j: usize) -> Result<isize, BandedError> {
        let bi = self.block_row_of(i)? as isize;
        let bj = self.block_col_of(j)? as isize;
        Ok(bj - bi)
    }
}
pub fn is_block_tridiagonal_position(
    layout: &NodeMajorLayout,
    i: usize,
    j: usize,
) -> Result<bool, BandedError> {
    Ok(layout.block_offset_of(i, j)?.abs() <= 1)
}

pub fn is_block_banded_position(
    layout: &NodeMajorLayout,
    i: usize,
    j: usize,
    block_half_bandwidth: usize,
) -> Result<bool, BandedError> {
    Ok(layout.block_offset_of(i, j)?.abs() <= block_half_bandwidth as isize)
}
#[cfg(test)]
mod tests {
    use super::NodeMajorLayout;

    #[test]
    fn global_index_roundtrip() {
        let layout = NodeMajorLayout::new(100, 50).unwrap();

        let g = layout.global_index(3, 17).unwrap();
        assert_eq!(g, 3 * 50 + 17);

        let (node, var) = layout.split_global(g).unwrap();
        assert_eq!(node, 3);
        assert_eq!(var, 17);
    }

    #[test]
    fn block_metadata_is_correct() {
        let layout = NodeMajorLayout::new(100, 50).unwrap();

        assert_eq!(layout.n_blocks(), 100);
        assert_eq!(layout.block_size(), 50);
        assert_eq!(layout.n_total(), 5000);
    }

    #[test]
    fn block_offsets_match_expectations() {
        let layout = NodeMajorLayout::new(4, 3).unwrap();

        let i = layout.global_index(1, 2).unwrap(); // node 1
        let j_same = layout.global_index(1, 0).unwrap(); // node 1
        let j_next = layout.global_index(2, 1).unwrap(); // node 2
        let j_prev = layout.global_index(0, 2).unwrap(); // node 0

        assert_eq!(layout.block_offset_of(i, j_same).unwrap(), 0);
        assert_eq!(layout.block_offset_of(i, j_next).unwrap(), 1);
        assert_eq!(layout.block_offset_of(i, j_prev).unwrap(), -1);
    }
}

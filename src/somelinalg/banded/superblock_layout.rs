use super::error::BandedError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuperBlockLayout {
    n_nodes: usize,
    vars_per_node: usize,
    nodes_per_superblock: usize,
}

impl SuperBlockLayout {
    pub fn new(
        n_nodes: usize,
        vars_per_node: usize,
        nodes_per_superblock: usize,
    ) -> Result<Self, BandedError> {
        if n_nodes == 0 || vars_per_node == 0 || nodes_per_superblock == 0 {
            return Err(BandedError::DimensionMismatch);
        }

        Ok(Self {
            n_nodes,
            vars_per_node,
            nodes_per_superblock,
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
    pub fn nodes_per_superblock(&self) -> usize {
        self.nodes_per_superblock
    }

    #[inline]
    pub fn superblock_size(&self) -> usize {
        self.vars_per_node * self.nodes_per_superblock
    }

    #[inline]
    pub fn n_total(&self) -> usize {
        self.n_nodes * self.vars_per_node
    }

    #[inline]
    pub fn n_superblocks(&self) -> usize {
        self.n_nodes.div_ceil(self.nodes_per_superblock)
    }

    /// Returns `true` when every superblock has the same full size.
    ///
    /// For the first diagnostics we keep the stricter "no tail block" contract,
    /// because the native block-tridiagonal solver currently assumes a uniform
    /// block size for every block.
    #[inline]
    pub fn is_evenly_divisible(&self) -> bool {
        self.n_nodes % self.nodes_per_superblock == 0
    }

    /// Alias kept for call sites that conceptually treat superblocks as the
    /// solver's blocks.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.superblock_size()
    }

    /// Alias kept for call sites that conceptually treat superblocks as the
    /// solver's blocks.
    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_superblocks()
    }

    #[inline]
    pub fn global_index(&self, node: usize, local_var: usize) -> Result<usize, BandedError> {
        if node >= self.n_nodes || local_var >= self.vars_per_node {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(node * self.vars_per_node + local_var)
    }

    #[inline]
    pub fn split_global(&self, global: usize) -> Result<(usize, usize), BandedError> {
        if global >= self.n_total() {
            return Err(BandedError::DimensionMismatch);
        }
        Ok((global / self.vars_per_node, global % self.vars_per_node))
    }

    #[inline]
    pub fn superblock_of_node(&self, node: usize) -> Result<usize, BandedError> {
        if node >= self.n_nodes {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(node / self.nodes_per_superblock)
    }

    #[inline]
    pub fn superblock_of_global(&self, global: usize) -> Result<usize, BandedError> {
        let (node, _) = self.split_global(global)?;
        self.superblock_of_node(node)
    }

    #[inline]
    pub fn local_node_in_superblock(&self, node: usize) -> Result<usize, BandedError> {
        if node >= self.n_nodes {
            return Err(BandedError::DimensionMismatch);
        }
        Ok(node % self.nodes_per_superblock)
    }

    #[inline]
    pub fn local_index_in_superblock(
        &self,
        node: usize,
        local_var: usize,
    ) -> Result<usize, BandedError> {
        if node >= self.n_nodes || local_var >= self.vars_per_node {
            return Err(BandedError::DimensionMismatch);
        }

        let local_node = node % self.nodes_per_superblock;
        Ok(local_node * self.vars_per_node + local_var)
    }

    #[inline]
    pub fn superblock_offset_of(&self, i: usize, j: usize) -> Result<isize, BandedError> {
        let bi = self.superblock_of_global(i)? as isize;
        let bj = self.superblock_of_global(j)? as isize;
        Ok(bj - bi)
    }
}

#[cfg(test)]
mod tests {
    use super::SuperBlockLayout;

    #[test]
    fn metadata_matches_superblock_grouping() {
        let layout = SuperBlockLayout::new(12, 6, 3).unwrap();

        assert_eq!(layout.n_nodes(), 12);
        assert_eq!(layout.vars_per_node(), 6);
        assert_eq!(layout.nodes_per_superblock(), 3);
        assert_eq!(layout.block_size(), 18);
        assert_eq!(layout.n_blocks(), 4);
        assert!(layout.is_evenly_divisible());
    }

    #[test]
    fn reports_tail_superblock_when_grouping_is_not_even() {
        let layout = SuperBlockLayout::new(10, 6, 4).unwrap();

        assert_eq!(layout.n_superblocks(), 3);
        assert!(!layout.is_evenly_divisible());
    }
}

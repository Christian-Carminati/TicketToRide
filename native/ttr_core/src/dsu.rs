//! Fast Disjoint Set Union (DSU) on stack-allocated arrays with path compression.

pub const MAX_CITIES: usize = 36;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisjointSet {
    pub parent: [u8; MAX_CITIES],
    pub rank: [u8; MAX_CITIES],
}

impl DisjointSet {
    pub fn new() -> Self {
        let mut parent = [0u8; MAX_CITIES];
        for i in 0..MAX_CITIES {
            parent[i] = i as u8;
        }
        DisjointSet {
            parent,
            rank: [0u8; MAX_CITIES],
        }
    }

    #[inline(always)]
    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while (root as u8) != self.parent[root] {
            root = self.parent[root] as usize;
        }
        let mut curr = i;
        while curr != root {
            let nxt = self.parent[curr] as usize;
            self.parent[curr] = root as u8;
            curr = nxt;
        }
        root
    }

    #[inline(always)]
    pub fn find_readonly(&self, mut i: usize) -> usize {
        while (i as u8) != self.parent[i] {
            i = self.parent[i] as usize;
        }
        i
    }

    #[inline(always)]
    pub fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if self.rank[root_i] < self.rank[root_j] {
                self.parent[root_i] = root_j as u8;
            } else if self.rank[root_i] > self.rank[root_j] {
                self.parent[root_j] = root_i as u8;
            } else {
                self.parent[root_j] = root_i as u8;
                self.rank[root_i] += 1;
            }
        }
    }

    #[inline(always)]
    pub fn is_connected(&self, i: usize, j: usize) -> bool {
        self.find_readonly(i) == self.find_readonly(j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsu_operations() {
        let mut dsu = DisjointSet::new();
        assert!(!dsu.is_connected(0, 1));
        dsu.union(0, 1);
        assert!(dsu.is_connected(0, 1));
        assert!(!dsu.is_connected(0, 2));
        dsu.union(1, 2);
        assert!(dsu.is_connected(0, 2));
    }
}

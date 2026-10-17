import torch


class ComplexMatrix:

    def normalizeComplexMatrix(self,complexmatrix,rule = "mulrank"):
        r,i = self.splitmatrix(complexmatrix)
        #print("r",r)
        r_ranked = self.getRankedMatrix(r)
        i_ranked = self.getRankedMatrix(i)
        #print("r_ranked",r_ranked,"max",torch.max(r_ranked))
        r_norm = self.normalizeMatrix(r_ranked)
        i_norm = self.normalizeMatrix(i_ranked)
        #print("r_norm",r_norm)
        if rule == "add":
            composedmatrix = self.normalizeMatrix(r_norm + i_norm)
            return composedmatrix
        if rule == "mulrank":
            return self.normalizeMatrix(r_ranked*i_ranked)

    def getRankedMatrix(self,matrix):
        # Get unique values and sort them
        unique_vals, inverse_indices = torch.unique(matrix, sorted=True, return_inverse=True)
        ranked_matrix = inverse_indices.view(matrix.shape)
        return ranked_matrix
    def normalizeMatrix(self,rankmatrix):
        return rankmatrix / torch.max(rankmatrix)

    def splitmatrix(self,complexmatrix):
        return complexmatrix.real,complexmatrix.imag




class TensorUtils:

    @staticmethod
    def split_positive_negative(A):
        """
            A+ is the positive part of A, A+ = max(A, 0)
            A- is the negative part of A, A- = min(A, 0)
            A = A+ + A-
        """
        A_pos = A.clone()
        A_pos[A < 0] = 0
        A_neg = A.clone()
        A_neg[A > 0] = 0
        return A_pos, A_neg



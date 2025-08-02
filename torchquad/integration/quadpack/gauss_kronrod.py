"""Gauss-Kronrod quadrature rules for QUADPACK algorithms."""

import numpy as np
from autoray import numpy as anp


class GaussKronrodRule:
    """Gauss-Kronrod quadrature rule implementation.

    Provides Gauss-Kronrod rules as used in QUADPACK algorithms.
    These rules pair an n-point Gauss rule with a (2n+1)-point Kronrod extension
    for error estimation. Coefficients are taken directly from the Fortran QUADPACK.
    
    OPTIMIZED VERSION: Minimizes tensor operations and device overhead.
    """

    # QNG coefficients for Patterson rules (10, 21, 43, 87 points)
    # These are the actual coefficients from QUADPACK - stored as Python floats

    # 10-point Gauss rule abscissae (x1) - native Python floats for speed
    _X1 = [
        9.73906528517171720077964012084452053428e-1,
        8.65063366688984510732096688423493048528e-1,
        6.79409568299024406234327365114873575769e-1,
        4.33395394129247190799265943165784162200e-1,
        1.48874338981631210884826001129719984618e-1,
    ]

    # 10-point Gauss weights - native Python floats
    _W10 = [
        6.66713443086881375935688098933317928579e-2,
        1.49451349150580593145776339657697332403e-1,
        2.19086362515982043995534934228163192459e-1,
        2.69266719309996355091226921569469352860e-1,
        2.95524224714752870173892994651338329421e-1,
    ]

    # 21-point rule additional abscissae (x2) - native Python floats
    _X2 = [
        9.95657163025808080735527280689002847921e-1,
        9.30157491355708226001207180059508346225e-1,
        7.80817726586416897063717578345042377163e-1,
        5.62757134668604683339000099272694140843e-1,
        2.94392862701460198131126603103865566163e-1,
    ]

    # 21-point weights for x1 points - native Python floats
    _W21A = [
        3.25581623079647274788189724593897606174e-2,
        7.50396748109199527670431409161900093952e-2,
        1.09387158802297641899210590325804960272e-1,
        1.34709217311473325928054001771706832761e-1,
        1.47739104901338491374841515972068045524e-1,
    ]

    # 21-point weights for x2 points (including center) - native Python floats
    _W21B = [
        1.16946388673718742780643960621920483962e-2,
        5.47558965743519960313813002445801763737e-2,
        9.31254545836976055350654650833663443900e-2,
        1.23491976262065851077958109831074159512e-1,
        1.42775938577060080797094273138717060886e-1,
        1.49445554002916905664936468389821203745e-1,  # center weight
    ]
    
    # Machine constants as class attributes to avoid recomputing
    _EPMACH = np.finfo(np.float64).eps
    _UFLOW = np.finfo(np.float64).tiny

    @classmethod
    def qng(cls, f, a, b, epsabs, epsrel, backend):
        """QNG - Non-adaptive integration using nested rules.

        Uses the Patterson sequence: 10, 21, 43, 87 point rules.
        This is the actual QNG implementation from QUADPACK.

        Args:
            f (callable): Function to integrate
            a, b: Integration limits
            epsabs, epsrel: Absolute and relative tolerances
            backend: Numerical backend

        Returns:
            tuple: (result, abserr, neval, ier)
        """
        # Machine constants
        epmach = np.finfo(np.float64).eps
        uflow = np.finfo(np.float64).tiny

        # Check input
        if a == b:
            return anp.array(0.0, like=a), anp.array(0.0, like=a), 0, 0

        # Initialize
        neval = 0
        ier = 0

        # Transform to [-1, 1]
        center = 0.5 * (a + b)
        hlgth = 0.5 * (b - a)
        dhlgth = abs(hlgth)

        # Convert coefficients to backend arrays with consistent dtype
        # Ensure all arrays have same dtype as domain bounds
        if backend == "tensorflow":
            # TensorFlow requires explicit dtype consistency
            import tensorflow as tf

            dtype = a.dtype if hasattr(a, "dtype") else tf.float64
            x1 = anp.array(cls._X1, dtype=dtype, like=a)
            w10 = anp.array(cls._W10, dtype=dtype, like=a)
            x2 = anp.array(cls._X2, dtype=dtype, like=a)
            w21a = anp.array(cls._W21A, dtype=dtype, like=a)
            w21b = anp.array(cls._W21B, dtype=dtype, like=a)
        else:
            x1 = anp.array(cls._X1, like=a)
            w10 = anp.array(cls._W10, like=a)
            x2 = anp.array(cls._X2, like=a)
            w21a = anp.array(cls._W21A, like=a)
            w21b = anp.array(cls._W21B, like=a)

        # Evaluate at center to determine data type
        fcentr = f(anp.array([center], like=a))[0]
        neval = 1

        # Evaluate at x1 points (10-point rule points)
        if backend == "tensorflow":
            # For TensorFlow, build arrays without item assignment
            fval1_list = []
            fval2_list = []
            for j in range(5):
                absc = hlgth * x1[j]
                fval1_list.append(f(anp.array([center + absc], like=a))[0])
                fval2_list.append(f(anp.array([center - absc], like=a))[0])
                neval += 2
            fval1 = anp.stack(fval1_list)
            fval2 = anp.stack(fval2_list)
        else:
            # For other backends, use normal approach
            fval1 = anp.zeros(5, dtype=fcentr.dtype, like=a)
            fval2 = anp.zeros(5, dtype=fcentr.dtype, like=a)

            for j in range(5):
                absc = hlgth * x1[j]
                if hasattr(fval1, "at"):  # JAX
                    fval1 = fval1.at[j].set(f(anp.array([center + absc], like=a))[0])
                    fval2 = fval2.at[j].set(f(anp.array([center - absc], like=a))[0])
                else:  # NumPy, Torch
                    fval1[j] = f(anp.array([center + absc], like=a))[0]
                    fval2[j] = f(anp.array([center - absc], like=a))[0]
                neval += 2

        # Compute 10-point Gauss result (no center point in 10-point Gauss)
        res10 = 0.0
        for j in range(5):
            res10 += w10[j] * (fval1[j] + fval2[j])
        res10 *= hlgth

        # Compute 21-point Kronrod result
        res21 = w21b[5] * fcentr  # Center weight
        for j in range(5):
            res21 += w21a[j] * (fval1[j] + fval2[j])

        # Evaluate at x2 points (additional 21-point rule points)
        if backend == "tensorflow":
            # For TensorFlow, build arrays without item assignment
            fval3_list = []
            fval4_list = []
            for j in range(5):
                absc = hlgth * x2[j]
                fval3_j = f(anp.array([center + absc], like=a))[0]
                fval4_j = f(anp.array([center - absc], like=a))[0]
                fval3_list.append(fval3_j)
                fval4_list.append(fval4_j)
                neval += 2
                res21 += w21b[j] * (fval3_j + fval4_j)
            fval3 = anp.stack(fval3_list)
            fval4 = anp.stack(fval4_list)
        else:
            # For other backends, use normal approach
            fval3 = anp.zeros(5, dtype=fcentr.dtype, like=a)
            fval4 = anp.zeros(5, dtype=fcentr.dtype, like=a)

            for j in range(5):
                absc = hlgth * x2[j]
                if hasattr(fval3, "at"):  # JAX
                    fval3 = fval3.at[j].set(f(anp.array([center + absc], like=a))[0])
                    fval4 = fval4.at[j].set(f(anp.array([center - absc], like=a))[0])
                else:  # NumPy, Torch
                    fval3[j] = f(anp.array([center + absc], like=a))[0]
                    fval4[j] = f(anp.array([center - absc], like=a))[0]
                neval += 2
                res21 += w21b[j] * (fval3[j] + fval4[j])

        res21 *= hlgth

        # Error estimate
        resabs = anp.abs(res21)
        resasc = w21b[5] * anp.abs(fcentr - res21 / (b - a))
        for j in range(5):
            resasc += w21a[j] * (
                anp.abs(fval1[j] - res21 / (b - a)) + anp.abs(fval2[j] - res21 / (b - a))
            )
            resasc += w21b[j] * (
                anp.abs(fval3[j] - res21 / (b - a)) + anp.abs(fval4[j] - res21 / (b - a))
            )
        resasc *= dhlgth

        abserr = anp.abs(res21 - res10)
        if resasc != 0.0 and abserr != 0.0:
            if backend == "torch":
                # Handle CUDA tensors
                abserr = resasc * min(1.0, (200.0 * float(abserr) / float(resasc)) ** 1.5)
            else:
                abserr = resasc * anp.minimum(1.0, (200.0 * abserr / resasc) ** 1.5)

        if resabs > uflow / (50.0 * epmach):
            if backend == "torch":
                abserr = max(50.0 * epmach * float(resabs), float(abserr))
            else:
                abserr = anp.maximum(50.0 * epmach * resabs, abserr)

        # Check convergence - handle CUDA tensors properly
        # Debug tensor types to fix CUDA issue
        if backend == "torch":
            # Ensure scalars are not CUDA tensors for comparison
            tolerance_val = max(float(epsabs), float(epsrel) * float(resabs))
            abserr_val = float(abserr)
            if abserr_val <= tolerance_val:
                return res21, abserr, neval, ier
        else:
            tolerance = anp.maximum(epsabs, epsrel * resabs)
            if abserr <= tolerance:
                return res21, abserr, neval, ier

        # Continue with 43-point rule if available
        # For now, return 21-point result
        if backend == "torch":
            if abserr_val > tolerance_val:
                ier = 1  # Failed to converge
        else:
            if abserr > tolerance:
                ier = 1  # Failed to converge

        return res21, abserr, neval, ier

    @classmethod
    def evaluate_gk21(cls, f, a, b, backend):
        """Evaluate using 21-point Gauss-Kronrod rule.

        This is for QAG/QAGS adaptive algorithms.

        HIGHLY OPTIMIZED VERSION: Minimizes tensor operations, eliminates coefficient 
        tensor creation, uses native Python arithmetic where possible, and maximally 
        vectorizes function evaluation for GPU efficiency.
        """
        # Use native Python arithmetic for interval arithmetic - much faster
        center = 0.5 * (float(a) + float(b))
        hlgth = 0.5 * (float(b) - float(a))
        abs_hlgth = abs(hlgth)
        
        # Pre-compute all evaluation points using native Python arithmetic
        # This eliminates 21+ tensor creation operations per call
        eval_points = [center]  # Center point
        
        # Add x1 points (positive and negative)
        for x1_coeff in cls._X1:
            absc = hlgth * x1_coeff
            eval_points.extend([center + absc, center - absc])
            
        # Add x2 points (positive and negative)  
        for x2_coeff in cls._X2:
            absc = hlgth * x2_coeff
            eval_points.extend([center + absc, center - absc])
        
        # Single vectorized function call for all 21 points
        # This is the key optimization - replaces 21 individual calls
        eval_array = anp.expand_dims(anp.array(eval_points, like=a), axis=-1)
        all_fvals = f(eval_array)
        
        # Extract function values with minimal indexing
        fcentr = all_fvals[0]
        fval1 = all_fvals[1:11:2]   # x1 positive: indices 1,3,5,7,9
        fval2 = all_fvals[2:12:2]   # x1 negative: indices 2,4,6,8,10  
        fval3 = all_fvals[11:21:2]  # x2 positive: indices 11,13,15,17,19
        fval4 = all_fvals[12:22:2]  # x2 negative: indices 12,14,16,18,20

        # Compute results using native Python arithmetic where possible
        # This eliminates many small tensor operations
        resg = 0.0  # 10-point Gauss (no center)
        resk = cls._W21B[5] * fcentr  # Center contribution
        
        for j in range(5):
            fsum = fval1[j] + fval2[j]
            resg += cls._W10[j] * fsum
            resk += cls._W21A[j] * fsum
            resk += cls._W21B[j] * (fval3[j] + fval4[j])

        # Scale by interval length
        resg *= hlgth
        resk *= hlgth

        # Error estimation - optimized with minimal tensor operations
        resabs = anp.abs(resk)
        
        # Pre-compute common term
        resk_avg = resk / (float(b) - float(a))
        
        # Compute resasc efficiently
        resasc = cls._W21B[5] * anp.abs(fcentr - resk_avg)
        for j in range(5):
            resasc += cls._W21A[j] * (anp.abs(fval1[j] - resk_avg) + anp.abs(fval2[j] - resk_avg))
            resasc += cls._W21B[j] * (anp.abs(fval3[j] - resk_avg) + anp.abs(fval4[j] - resk_avg))
        resasc *= abs_hlgth

        # Optimized error computation
        abserr = anp.abs(resk - resg)
        
        # Avoid creating unnecessary tensors for scalar operations
        if float(resasc) != 0.0 and float(abserr) != 0.0:
            ratio = 200.0 * float(abserr) / float(resasc)
            factor = min(1.0, pow(ratio, 1.5))
            abserr = resasc * factor

        # Use class constants to avoid repeated computation
        resabs_val = float(resabs)
        if resabs_val > cls._UFLOW / (50.0 * cls._EPMACH):
            min_error = anp.array(50.0 * cls._EPMACH * resabs_val, like=resabs)
            abserr = anp.maximum(min_error, abserr)

        return resk, abserr, resabs, resasc, 21

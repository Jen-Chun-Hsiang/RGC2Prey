import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any
import warnings

@dataclass
class LNKParams:
    """Parameters for the LNK rate model."""
    tau: float
    alpha_d: float
    sigma0: float
    alpha: float
    beta: float
    b_out: float
    g_out: float
    theta: float

class LNKRateModel:
    """
    1-state LNK (additive + multiplicative) model for continuous firing-rate traces.
    
    Model (discrete time, single kinetic state):
      a_{t+1} = a_t + dt * (alpha_d * F(x_t) - a_t) / tau,  with F(x) = max(0, x - theta)
      ỹ_t     = x_t / (sigma0 + alpha * a_t) + beta * a_t + b_out
      r_t     = outNL( g_out * ỹ_t )   (default outNL = softplus)
    """
    
    def __init__(self):
        self.fitted_params = None
    
    def softplus(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable softplus function."""
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    
    def _default_init(self, x: np.ndarray, y_rate: np.ndarray, dt: float) -> LNKParams:
        """Generate default initial parameters."""
        ypos = np.maximum(y_rate, 0)
        rx = np.mean(np.abs(x[x != 0])) if np.any(x != 0) else 1.0
        ry = np.mean(ypos[ypos > 0]) if np.any(ypos > 0) else 1.0
        
        return LNKParams(
            tau=np.random.rand(),
            alpha_d=np.random.rand(),
            theta=max(0, np.percentile(x, 20)),
            sigma0=0.1 + 0.05 * np.std(x),
            alpha=0.01,
            beta=-0.1,
            b_out=1.0,
            g_out=max(ry / (rx + np.finfo(float).eps), 0.5)
        )
    
    def _pack_params(self, params: LNKParams) -> np.ndarray:
        """Pack parameters into optimization vector."""
        return np.array([
            np.log(max(params.tau, 1e-6)),
            np.log(max(params.alpha_d, 1e-6)),
            np.log(max(params.sigma0, 1e-9)),
            np.log(max(params.alpha, 1e-9)),
            params.beta,
            params.b_out,
            np.log(max(params.g_out, 1e-9)),
            np.log(max(params.theta, 0) + 1)  # softplus inverse approx
        ])
    
    def _unpack_params(self, p: np.ndarray) -> LNKParams:
        """Unpack optimization vector into parameters."""
        return LNKParams(
            tau=np.exp(p[0]),
            alpha_d=np.exp(p[1]),
            sigma0=np.exp(p[2]),
            alpha=np.exp(p[3]),
            beta=p[4],
            b_out=p[5],
            g_out=np.exp(p[6]),
            theta=self.softplus(p[7])
        )
    
    def forward_pass(self, x: np.ndarray, params: LNKParams, dt: float, 
                    output_nl: str = 'softplus') -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the LNK model.
        
        Args:
            x: Input drive signal (T,)
            params: Model parameters
            dt: Time step in seconds
            output_nl: Output nonlinearity ('softplus' or 'linear')
            
        Returns:
            rate: Predicted firing rate (T,)
            a: Kinetic state trajectory (T,)
        """
        x = np.asarray(x).flatten()
        T = len(x)
        a = np.zeros(T)
        
        # Forward dynamics
        for t in range(T - 1):
            drive = max(0, x[t] - params.theta)
            a[t + 1] = a[t] + dt * (params.alpha_d * drive - a[t]) / params.tau
            if a[t + 1] < 0:
                a[t + 1] = 0
        
        # Output computation
        den = params.sigma0 + params.alpha * a
        den = np.maximum(den, 1e-9)  # Avoid division by zero
        add = params.beta * a
        ytilde = x / den + add + params.b_out
        
        if output_nl.lower() == 'softplus':
            rate = self.softplus(params.g_out * ytilde)
        elif output_nl.lower() == 'linear':
            rate = np.maximum(params.g_out * ytilde, 0)  # Clamp to non-negative
        else:
            raise ValueError("output_nl must be 'softplus' or 'linear'")
        
        return rate, a
    
    def _objective(self, p: np.ndarray, x: np.ndarray, y_rate: np.ndarray, 
                  dt: float, weights: np.ndarray, robust: str, delta: float, 
                  output_nl: str, ridge: float) -> float:
        """Objective function for optimization."""
        try:
            params = self._unpack_params(p)
            rate_pred, _ = self.forward_pass(x, params, dt, output_nl)
            
            residuals = rate_pred - y_rate
            
            if robust.lower() == 'none':
                loss = np.mean(weights * (residuals ** 2))
            elif robust.lower() == 'huber':
                abs_res = np.abs(residuals)
                huber = np.where(abs_res <= delta,
                               0.5 * residuals ** 2,
                               delta * (abs_res - 0.5 * delta))
                loss = np.mean(weights * huber)
            else:
                raise ValueError("robust must be 'none' or 'huber'")
            
            # L2 regularization
            loss += ridge * np.sum(p ** 2)
            
            return loss
            
        except Exception as e:
            # Return large loss if computation fails
            return 1e10
    
    def fit(self, x: np.ndarray, y_rate: np.ndarray, dt: float,
            init_params: Optional[Dict[str, Any]] = None,
            max_iter: int = 400,
            weights: Optional[np.ndarray] = None,
            robust: str = 'none',
            delta: float = 1.0,
            output_nl: str = 'softplus',
            ridge: float = 0.0) -> Tuple[LNKParams, np.ndarray, np.ndarray, float]:
        """
        Fit the LNK rate model to data.
        
        Args:
            x: Input drive signal (T,)
            y_rate: Measured firing rate (T,)
            dt: Time step in seconds
            init_params: Initial parameter values (optional)
            max_iter: Maximum optimization iterations
            weights: Per-sample weights for MSE (optional)
            robust: Robust fitting ('none' or 'huber')
            delta: Huber delta parameter
            output_nl: Output nonlinearity ('softplus' or 'linear')
            ridge: L2 regularization strength
            
        Returns:
            params: Fitted parameters
            rate_hat: Fitted rate trajectory
            a_traj: Fitted kinetic state trajectory
            fval: Final objective value
        """
        x = np.asarray(x).flatten()
        y_rate = np.asarray(y_rate).flatten()
        T = len(x)
        
        if len(y_rate) != T:
            raise ValueError("x and y_rate must have the same length")
        
        if weights is None:
            weights = np.ones(T)
        else:
            weights = np.asarray(weights).flatten()
            if len(weights) != T:
                raise ValueError("weights must have the same length as x and y_rate")
        
        # Initialize parameters
        init = self._default_init(x, y_rate, dt)
        if init_params is not None:
            for key, value in init_params.items():
                if hasattr(init, key):
                    setattr(init, key, value)
        
        p0 = self._pack_params(init)
        
        # Parameter bounds
        lb = np.array([np.log(1e-3), np.log(1e-6), np.log(1e-6), np.log(1e-6), 
                      -10, -10, np.log(1e-6), np.log(1)])
        ub = np.array([np.log(10), np.log(10), np.log(10), np.log(10), 
                      10, 10, np.log(100), np.log(max(np.max(x), 0) + 1)])
        
        bounds = list(zip(lb, ub))
        
        # Optimize
        objective = lambda p: self._objective(p, x, y_rate, dt, weights, 
                                            robust, delta, output_nl, ridge)
        
        result = minimize(objective, p0, method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': max_iter, 'disp': False})
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Unpack final parameters and compute outputs
        self.fitted_params = self._unpack_params(result.x)
        rate_hat, a_traj = self.forward_pass(x, self.fitted_params, dt, output_nl)
        
        return self.fitted_params, rate_hat, a_traj, result.fun
    
    def predict(self, x: np.ndarray, dt: float, params: Optional[LNKParams] = None,
               output_nl: str = 'softplus') -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast inference: predict firing rate given input and fitted parameters.
        
        Args:
            x: Input drive signal (T,)
            dt: Time step in seconds
            params: Model parameters (uses fitted_params if None)
            output_nl: Output nonlinearity ('softplus' or 'linear')
            
        Returns:
            rate_hat: Predicted firing rate (T,)
            a_traj: Kinetic state trajectory (T,)
        """
        if params is None:
            if self.fitted_params is None:
                raise ValueError("No fitted parameters available. Call fit() first or provide params.")
            params = self.fitted_params
        
        return self.forward_pass(x, params, dt, output_nl)

# Convenience function to match MATLAB interface
def fit_lnk_rate(x: np.ndarray, y_rate: np.ndarray, dt: float, **kwargs) -> Tuple[LNKParams, np.ndarray, np.ndarray, float]:
    """
    Convenience function that matches the MATLAB fitLNK_rate interface.
    
    Args:
        x: Input drive signal (T,)
        y_rate: Measured firing rate (T,)
        dt: Time step in seconds
        **kwargs: Additional options (init_params, max_iter, weights, robust, delta, output_nl, ridge)
        
    Returns:
        params: Fitted parameters
        rate_hat: Fitted rate trajectory
        a_traj: Fitted kinetic state trajectory
        fval: Final objective value
    """
    model = LNKRateModel()
    return model.fit(x, y_rate, dt, **kwargs)

# Fast inference function
def predict_lnk_rate(x: np.ndarray, params: LNKParams, dt: float, 
                    output_nl: str = 'softplus') -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast inference function for LNK rate prediction.
    
    Args:
        x: Input drive signal (T,)
        params: Fitted model parameters
        dt: Time step in seconds
        output_nl: Output nonlinearity ('softplus' or 'linear')
        
    Returns:
        rate_hat: Predicted firing rate (T,)
        a_traj: Kinetic state trajectory (T,)
    """
    model = LNKRateModel()
    return model.forward_pass(x, params, dt, output_nl)

# Example usage
"""
    # Generate synthetic data
    np.random.seed(42)
    T = 1000
    dt = 0.001  # 1ms bins
    t = np.arange(T) * dt
    
    # Create synthetic drive signal
    x = 2 * np.sin(2 * np.pi * t) + np.random.randn(T) * 0.1
    
    # Generate synthetic firing rate (for demo)
    true_params = LNKParams(tau=0.1, alpha_d=1.5, sigma0=0.5, alpha=0.1, 
                           beta=0.2, b_out=0.5, g_out=10.0, theta=0.0)
    model = LNKRateModel()
    y_true, _ = model.forward_pass(x, true_params, dt)
    y_rate = y_true + np.random.randn(T) * 0.5  # Add noise
    
    # Fit the model
    print("Fitting LNK rate model...")
    fitted_params, rate_hat, a_traj, fval = fit_lnk_rate(x, y_rate, dt, max_iter=100)
    
    print(f"Final objective value: {fval:.4f}")
    print(f"Fitted parameters:")
    print(f"  tau: {fitted_params.tau:.4f}")
    print(f"  alpha_d: {fitted_params.alpha_d:.4f}")
    print(f"  theta: {fitted_params.theta:.4f}")
    print(f"  sigma0: {fitted_params.sigma0:.4f}")
    print(f"  alpha: {fitted_params.alpha:.4f}")
    print(f"  beta: {fitted_params.beta:.4f}")
    print(f"  b_out: {fitted_params.b_out:.4f}")
    print(f"  g_out: {fitted_params.g_out:.4f}")
    
    # Fast inference on new data
    print("\nTesting fast inference...")
    x_new = np.random.randn(500)
    rate_pred, a_pred = predict_lnk_rate(x_new, fitted_params, dt)
    print(f"Predicted rate range: [{rate_pred.min():.2f}, {rate_pred.max():.2f}] Hz")
    
    # Using the class interface for multiple predictions
    print("\nUsing class interface...")
    model_fitted = LNKRateModel()
    model_fitted.fitted_params = fitted_params
    
    # Multiple fast predictions
    for i in range(3):
        x_test = np.random.randn(100)
        rate_test, _ = model_fitted.predict(x_test, dt)
        print(f"Test {i+1} - Mean predicted rate: {rate_test.mean():.2f} Hz")

"""
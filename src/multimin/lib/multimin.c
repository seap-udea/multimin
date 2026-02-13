#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Structure to hold pre-calculated decomposition and constants
 */
typedef struct {
  int k;
  gsl_vector *mu;
  gsl_matrix *L;              // Cholesky decomposition L of Sigma
  double normalization_const; // (2pi)^(-k/2) * |Sigma|^(-0.5)
} GaussianParams;

/*
 * Initialize parameters from raw arrays.
 * mu_arr: k elements
 * sigma_arr: k*k elements (row-major)
 */
int init_params(GaussianParams *p, int k, const double *mu_arr,
                const double *sigma_arr) {
  p->k = k;
  p->mu = gsl_vector_alloc(k);
  p->L = gsl_matrix_alloc(k, k);

  // Copy mu
  for (int i = 0; i < k; i++) {
    gsl_vector_set(p->mu, i, mu_arr[i]);
  }

  // Copy sigma to L (will be overwritten by Cholesky)
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      gsl_matrix_set(p->L, i, j, sigma_arr[i * k + j]);
    }
  }

  // Cholesky: Sigma = L L^T
  int status = gsl_linalg_cholesky_decomp(p->L);
  if (status != 0) {
    return status; // Error, e.g. not positive definite
  }

  // Calculate log(det(Sigma)) = 2 * sum(log(diag(L)))
  double log_det = 0.0;
  for (int i = 0; i < k; i++) {
    log_det += log(gsl_matrix_get(p->L, i, i));
  }
  log_det *= 2.0;

  // log_const = -k/2 * log(2pi) - 0.5 * log_det
  double log_const = -0.5 * k * log(2.0 * M_PI) - 0.5 * log_det;
  p->normalization_const = exp(log_const);
  return 0;
}

void free_params(GaussianParams *p) {
  if (p->mu)
    gsl_vector_free(p->mu);
  if (p->L)
    gsl_matrix_free(p->L);
}

/*
 * Core evaluation function for one point
 * x: vector of size k
 * work: pre-allocated vector of size k
 */
double eval_pdf(const GaussianParams *p, const gsl_vector *x,
                gsl_vector *work) {
  // work = x - mu
  gsl_vector_memcpy(work, x);
  gsl_vector_sub(work, p->mu);

  // Solve L * y = (x - mu). L is lower triangular.
  // Result y is stored in work.
  gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, p->L, work);

  // Mahalanobis distance squared = y . y
  double mahalanobis_sq;
  gsl_blas_ddot(work, work, &mahalanobis_sq);

  return p->normalization_const * exp(-0.5 * mahalanobis_sq);
}

/*
 * Batch NMD
 * X: (n_points * k) flattened array
 * mu: (k) array
 * sigma: (k * k) flattened array
 * results: (n_points) array
 */
void nmd_batch(double *X, int n_points, int k, double *mu, double *sigma,
               double *results) {
  GaussianParams params;
  if (init_params(&params, k, mu, sigma) != 0) {
    // Handle error: fill results with NAN
    for (int i = 0; i < n_points; i++)
      results[i] = NAN;
    return;
  }

  gsl_vector *work = gsl_vector_alloc(k);
  gsl_vector_view x_view;

  for (int i = 0; i < n_points; i++) {
    x_view = gsl_vector_view_array(&X[i * k], k);
    results[i] = eval_pdf(&params, &x_view.vector, work);
  }

  gsl_vector_free(work);
  free_params(&params);
}

/*
 * Batch Truncated NMD
 * lb: lower bounds (k) - pass NULL implemented as pointers in C logic?
 *     Actually better to pass an array. If caller wants no bound, pass
 * -INFINITY. ub: upper bounds (k) inv_Z: inverse normalization constant (1/Z)
 */
void tnmd_batch(double *X, int n_points, int k, double *mu, double *sigma,
                double *lb, double *ub, double inv_Z, double *results) {

  GaussianParams params;
  if (init_params(&params, k, mu, sigma) != 0) {
    for (int i = 0; i < n_points; i++)
      results[i] = NAN;
    return;
  }

  gsl_vector *work = gsl_vector_alloc(k);
  gsl_vector_view x_view;

  for (int i = 0; i < n_points; i++) {
    int out_of_bounds = 0;
    // Check bounds
    for (int j = 0; j < k; j++) {
      double val = X[i * k + j];
      // Check lower bound
      if (lb && val < lb[j]) {
        out_of_bounds = 1;
        break;
      }
      // Check upper bound
      if (ub && val > ub[j]) {
        out_of_bounds = 1;
        break;
      }
    }

    if (out_of_bounds) {
      results[i] = 0.0;
    } else {
      x_view = gsl_vector_view_array(&X[i * k], k);
      double pdf = eval_pdf(&params, &x_view.vector, work);
      results[i] = pdf * inv_Z;
    }
  }

  gsl_vector_free(work);
  free_params(&params);
}

/*
 * Helper: Solve L * y = b using forward substitution.
 * L is lower triangular k*k (row-major).
 * b is k-dim vector.
 * y is result k-dim vector.
 */
void forward_sub(int k, const double *L, const double *b, double *y) {
  for (int i = 0; i < k; i++) {
    double sum = 0.0;
    for (int j = 0; j < i; j++) {
      sum += L[i * k + j] * y[j];
    }
    y[i] = (b[i] - sum) / L[i * k + i];
  }
}

/*
 * Batch Mixture of Gaussians (MoG)
 * X: (n_points * k) flattened array
 * n_points: number of points
 * k: dimension
 * n_comps: number of components
 * weights: (n_comps) array
 * mus: (n_comps * k) flattened array
 * sigmas: (n_comps * k * k) flattened array
 * results: (n_points) array
 */
void mog_batch(double *X, int n_points, int k, int n_comps, double *weights,
               double *mus, double *sigmas, double *results) {

  // Allocate array of params
  GaussianParams *params_array =
      (GaussianParams *)malloc(n_comps * sizeof(GaussianParams));
  if (params_array == NULL) {
    // Memory allocation failed
    for (int i = 0; i < n_points; i++)
      results[i] = NAN;
    return;
  }

  // Initialize and extract raw pointers for L
  double **L_arrays = (double **)malloc(n_comps * sizeof(double *));
  if (L_arrays == NULL) {
    free(params_array);
    return;
  }

  for (int j = 0; j < n_comps; j++) {
    if (init_params(&params_array[j], k, &mus[j * k], &sigmas[j * k * k]) !=
        0) {
      for (int jj = 0; jj < j; jj++)
        free_params(&params_array[jj]);
      free(params_array);
      free(L_arrays);
      for (int i = 0; i < n_points; i++)
        results[i] = NAN;
      return;
    }
    // Get pointer to L data for fast access
    L_arrays[j] = gsl_matrix_ptr(params_array[j].L, 0, 0);
  }

  // Work arrays
  double *diff = (double *)malloc(k * sizeof(double));
  double *y = (double *)malloc(k * sizeof(double));

  // Process points
  for (int i = 0; i < n_points; i++) {
    double sum = 0.0;
    double *x_ptr = &X[i * k];

    for (int j = 0; j < n_comps; j++) {
      // inline evaluation
      // diff = x - mu
      gsl_vector *mu_vec = params_array[j].mu;
      for (int d = 0; d < k; d++) {
        diff[d] = x_ptr[d] - gsl_vector_get(mu_vec, d);
      }

      // Solve L * y = diff
      forward_sub(k, L_arrays[j], diff, y);

      // Mahalanobis = dot(y, y)
      double mah_sq = 0.0;
      for (int d = 0; d < k; d++)
        mah_sq += y[d] * y[d];

      sum +=
          weights[j] * params_array[j].normalization_const * exp(-0.5 * mah_sq);
    }
    results[i] = sum;
  }

  free(diff);
  free(y);
  free(L_arrays);
  for (int j = 0; j < n_comps; j++) {
    free_params(&params_array[j]);
  }
  free(params_array);
}

/*
 * Batch Truncated MoG
 * lb, ub: (k) arrays for boundaries (global)
 * inv_Zs: (n_comps) array of inverse normalization constants
 */
void tmog_batch(double *X, int n_points, int k, int n_comps, double *weights,
                double *mus, double *sigmas, double *lb, double *ub,
                double *inv_Zs, double *results) {

  GaussianParams *params_array =
      (GaussianParams *)malloc(n_comps * sizeof(GaussianParams));
  if (params_array == NULL) {
    for (int i = 0; i < n_points; i++)
      results[i] = NAN;
    return;
  }

  double **L_arrays = (double **)malloc(n_comps * sizeof(double *));
  if (L_arrays == NULL) {
    free(params_array);
    return;
  }

  for (int j = 0; j < n_comps; j++) {
    if (init_params(&params_array[j], k, &mus[j * k], &sigmas[j * k * k]) !=
        0) {
      for (int jj = 0; jj < j; jj++)
        free_params(&params_array[jj]);
      free(params_array);
      free(L_arrays);
      for (int i = 0; i < n_points; i++)
        results[i] = NAN;
      return;
    }
    L_arrays[j] = gsl_matrix_ptr(params_array[j].L, 0, 0);
  }

  double *diff = (double *)malloc(k * sizeof(double));
  double *y = (double *)malloc(k * sizeof(double));

  for (int i = 0; i < n_points; i++) {
    // Check bounds first (global)
    int out_of_bounds = 0;
    for (int d = 0; d < k; d++) {
      double val = X[i * k + d];
      if (lb && val < lb[d]) {
        out_of_bounds = 1;
        break;
      }
      if (ub && val > ub[d]) {
        out_of_bounds = 1;
        break;
      }
    }

    if (out_of_bounds) {
      results[i] = 0.0;
    } else {
      double sum = 0.0;
      double *x_ptr = &X[i * k];

      for (int j = 0; j < n_comps; j++) {
        // inline evaluation
        gsl_vector *mu_vec = params_array[j].mu;
        for (int d = 0; d < k; d++) {
          diff[d] = x_ptr[d] - gsl_vector_get(mu_vec, d);
        }

        forward_sub(k, L_arrays[j], diff, y);

        double mah_sq = 0.0;
        for (int d = 0; d < k; d++)
          mah_sq += y[d] * y[d];

        double pdf = params_array[j].normalization_const * exp(-0.5 * mah_sq);
        sum += weights[j] * pdf * inv_Zs[j];
      }
      results[i] = sum;
    }
  }

  free(diff);
  free(y);
  free(L_arrays);
  for (int j = 0; j < n_comps; j++) {
    free_params(&params_array[j]);
  }
  free(params_array);
}

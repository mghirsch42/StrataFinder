#define EIGEN_WARNINGS_DISABLED
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <numeric>
#include <utility>
#include <algorithm>
#include <limits>
#include <math.h>
#include <Rcpp.h>
using namespace std;
using namespace Rcpp;

// If fit type is exponential, return the log-transformed y values. Otherwise, return the original y values.
// @param y y values to transform
// @param fit_type the fit type of the model
// @return log-transformed y values or copy of original y values
Rcpp::NumericVector transform_y(Rcpp::NumericVector y, std::string fit_type) {
    Rcpp::NumericVector y_ (y.size());
    if (fit_type == "exponential") {
        transform(y.begin(), y.end(), y_.begin(), [](int i){
            if (i == 0) i = 1;
            return log(i);
        });
    }
    else {
        y_ = Rcpp::NumericVector(y.begin(), y.end());
    }
    return y_;
};

// Calculate the residuals for a single segment and parameter set
// @param x x values
// @param y y values
// @param intercept intercept value
// @param coefs list of coefficient values, ordered from lowest power of x to highest
// @param fit_type equation type to fit. Can be exponential, quadratic, flat, 
// @return a vector of the residuals for that segment
std::vector<double> single_segment_residuals(std::vector<int> x, std::vector<double> y, double intercept, std::vector<double> coefs, std::string fit_type) {
    int n = x.size();   // Number of elements
    std::vector<double> reds = std::vector<double>();   // List of residuals at each index

    double ypred;   // Predicted y value for index i
    double residual;    // Residual at index i

    // If fit_type is linear or exponential, coefs[0] is the coefficient to x
    // If fit_type is quadrative, coefs[0] is the coefficient to x, and coefs[1] is the coefficient to x^2
    // If fit_type is flat, coefs is unused.

    // Calculate residuals at each index appropriately depending on the fit type
    for (int i=0; i<n; i++) {
        if (fit_type == "exponential") {
            ypred = exp(intercept + coefs[0]*x[i]);
        } 
        else if (fit_type == "quadratic") {
            ypred = intercept + coefs[0]*x[i] + coefs[1]*x[i]*x[i];
        }
        else if (fit_type == "flat") {
            ypred = intercept;
        }
        else if (fit_type == "linear") {
            ypred = intercept + coefs[0]*x[i];
        }
        else {
            cout << "Unknown fit type. Fit type should be flat, linear, exponential, or quadratic." << endl;
            // exit(0);
            return reds;
        }
        residual = (y[i] - ypred) * (y[i] - ypred);
        if (isinf(residual)) {
            reds.push_back(std::numeric_limits<double>::max());
        }
        else {
            reds.push_back(residual);
        }
    }
    // Return residuals
    return reds;
};

// Use matrix decomposition to estimate a quadratic function to fit the given data
// @param x x values
// @param y original y values
// @param y_ transformed y values (the same as y, leaving so it's consistent with linear function)
std::tuple<double, std::vector<double>, std::vector<double>> single_segment_regression_quadratic(std::vector<int> x, std::vector<double> y, std::vector<double> y_) {
    int n = x.size();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 3); // First column is 1 for the coefficent, second column is each x value, third is x^2
    Eigen::VectorXd y_e = Eigen::VectorXd::Ones(n); // Eigen vector for y
    Eigen::VectorXd b; // [Intercept, Coefficient for x, Coefficient for x^2] (what we're going to estimate)

    // Initialize the second and third columns of the X matrix to each x value and each y value
    for (int i=0; i<n; i++) {
        X(i,1) = x[i];
        X(i,2) = x[i]*x[i];
        y_e(i) = y_[i];
    }
    b = X.colPivHouseholderQr().solve(y_e);
    double intercept = b(0);
    std::vector<double> coefs {b(1), b(2)};
    
    // Get the residuals for this segment
    std::vector<double> residuals = single_segment_residuals(x, y, intercept, coefs, "quadratic");
    // Return the m, b, and the residuals
    std::tuple<double, std::vector<double>, std::vector<double>> results {intercept, coefs, residuals};
    return results;
};

// Use matrix decomposition to estimate a linear function to fit the given data (used for linear and exponential fits)
// @param x x values
// @param y original y values (used for residuals)
// @param y_ transformed y values (log-transformed for exponential fits, the same as y for linear fits; used for equation estimation)
std::tuple<double, std::vector<double>, std::vector<double>> single_segment_regression_linear(std::vector<int> x, std::vector<double> y, std::vector<double> y_, std::string fit_type) {
    int n = x.size();
    Eigen::MatrixXd X = Eigen::MatrixXd::Ones(n, 2); // First column is 1 for the coefficent, second column is each x value
    Eigen::VectorXd y_e = Eigen::VectorXd::Ones(n); // Eigen vector for y
    Eigen::VectorXd b; // [Intercept, Coefficient] (what we're going to estimate)

    // Initialize the second column of the X matrix to each x value and each y value
    for (int i=0; i<n; i++) {
        X(i,0) = 1;
        X(i,1) = x[i];
        y_e(i) = y_[i];
    }
    b = X.colPivHouseholderQr().solve(y_e);
    double intercept = b(0);
    std::vector<double> coefs = {b(1)};
    
    // Get the residuals for this segment
    std::vector<double> residuals = single_segment_residuals(x, y, intercept, coefs, fit_type);

    // Return the m, b, and the residuals
    std::tuple<double, std::vector<double>, std::vector<double>> results {intercept, coefs, residuals};
    return results;
}

// Take the average value of the vector to be the intercept of a linear fit with zero slop.
// @param x x values
// @param y original y values
// @param y_ transformed y values (log-transformed for exponential fits, the same as y for linear fits)
std::tuple<double, std::vector<double>, std::vector<double>> single_segment_regression_flat(std::vector<int> x, std::vector<double> y) {
    int n = x.size();

    // The intercept is equal to the mean of the vector
    double intercept = accumulate(y.begin(), y.end(), 0) / n;
    // The coefficient is 0
    std::vector<double> coefs = {0};
    
    // Get the residuals for this segment
    std::vector<double> residuals = single_segment_residuals(x, y, intercept, coefs, "flat");

    // Return the m, b, and the residuals
    std::tuple<double, std::vector<double>, std::vector<double>> results {intercept, coefs, residuals};
    return results;
}

std::tuple<double, std::vector<double>, std::vector<double>> single_segment_regression(std::vector<int> x, std::vector<double> y, std::vector<double> y_, std::string fit_type) {
    if (fit_type == "quadratic") {
        return single_segment_regression_quadratic(x, y, y_);
    }
    if (fit_type == "flat") {
        return single_segment_regression_flat(x, y);
    }
    else {
        return single_segment_regression_linear(x, y, y_, fit_type);
    }
};

std::vector<std::vector<double>> base_case(int min_len, std::vector<std::vector<double>> ssr_mat) {
    int n = ssr_mat.size();
    int m = 1;
    std::vector<std::vector<double>> results (n-(m+1)*min_len+1, vector<double>(3, 0));

    for (int j=m*min_len; j<n-m*min_len+1; j+=1) {
        double left = ssr_mat[m-1][j];
        double right = ssr_mat[j+1][n-1];
        results[j-m*min_len] = {(double)j, left, right};
    }

    return results;
};

std::vector<std::vector<double>> recurse(int curr_n_bp, int min_len, std::vector<std::vector<double>> ssr_mat) {
    if (curr_n_bp == 1) {
        return base_case(min_len, ssr_mat);
    }
    int n = ssr_mat.size();
    int min_k;
    double min_err, last_err, left, right;
    std::vector<double> last;
    std::vector<std::vector<double>> last_results = recurse(curr_n_bp-1, min_len, ssr_mat);
    std::vector<std::vector<double>> results (n-(curr_n_bp+1)*min_len+1, vector<double>(2*curr_n_bp+1, 0));
    
    for (int j=curr_n_bp*min_len; j<n-min_len+1; j+=1) {
        min_k = (curr_n_bp-1)*min_len;
        min_err = std::numeric_limits<int>::max();
        for (int k=(curr_n_bp-1)*min_len; k<j-min_len+1; k+=1) {
            last_err = std::accumulate(last_results[k-(curr_n_bp-1)*min_len].begin()+curr_n_bp-1, last_results[k-(curr_n_bp-1)*min_len].end()-1, 0.0);
            std::vector<double> tmp;
            tmp.insert(tmp.begin(), last_results[k-(curr_n_bp-1)*min_len].begin()+curr_n_bp-1, last_results[k-(curr_n_bp-1)*min_len].end());
            left = ssr_mat[k+1][j];
            right = ssr_mat[j+1][n-1];
            if (last_err+left+right < min_err) {
                min_k = k;
                min_err = last_err+left+right;
            }
        }
        last = last_results[min_k-(curr_n_bp-1)*min_len];
        left = ssr_mat[min_k+1][j];
        right = ssr_mat[j+1][n-1];
        int a = j-curr_n_bp*min_len;
        for (int i=0; i<curr_n_bp-1; i++) {
            results[a][i] = last[i];
        }
        results[a][curr_n_bp-1] = j;
        for (unsigned long int i=curr_n_bp-1; i<last.size(); i++){
            results[a][i+1] = last[i];
        }
        results[a][results[a].size()-2] = left;
        results[a][results[a].size()-1] = right;
    }
    return results;
};

std::tuple<std::vector<int>, double> run(int curr_n_bp, int min_len, std::vector<std::vector<double>> ssr_mat){
    std::vector<std::vector<double>> results = recurse(curr_n_bp, min_len, ssr_mat);
    int min_idx = 0;
    double min_err = std::numeric_limits<int>::max();
    double new_err = 0;
    for (unsigned long int i=0; i<results.size(); i++) {
        new_err = std::accumulate(results[i].begin()+curr_n_bp, results[i].end(), 0.0);
        if (new_err < min_err) {
            min_idx = i;
            min_err = new_err;
        }
    }
    std::vector<int> bps;
    bps.insert(bps.begin(), results[min_idx].begin(), results[min_idx].begin()+curr_n_bp);
    return std::tuple<std::vector<int>, double> {bps, min_err};
};

//' Estimate the equation parameters and residuals for each piece of the data
//' @param x x values
//' @param y y values
//' @param bps breakpoint indices
//' @param fit_type type of equation to fit (linear, exponential, or quadratic)
//' @return a named list of the intercept, coefficients, sum squared residuals, and residual values for each segment
// [[Rcpp::export]]
Rcpp::List piecewise_regression(Rcpp::IntegerVector x, Rcpp::NumericVector y, Rcpp::IntegerVector bps, std::string fit_type) {
    // A note on the ranges for each regression segment: the vector initializer includes the first value and excludes
    // the last value. So the range x.begin(), x.begin()+bps[0]+1 includes 0 up to and including the first breakpoint.
    // The range x.begin()+bps[i], x.begin()+bps[i+1]+1 includes breakpoint i up to and including the next breakpoint.
    bps.push_front(0);
    bps.push_back(x.size()-1);
    Rcpp::NumericVector y_ = transform_y(y, fit_type);

    std::vector<double> ssrs ((int) bps.size()-1, 0);
    std::vector<double> intercept_list ((int) bps.size()-1, 0);
    std::vector<std::vector<double>> coefs_list ((int) bps.size()-1);
    std::vector<double> residuals ((int) bps.size()-1, 0);

    std::tuple<double, std::vector<double>, std::vector<double>> reg_results;
    std::vector<double> new_residuals;
    std::vector<double> new_coefs;

    for (int i=0; i<bps.size()-1; i++) {    
        reg_results = single_segment_regression(
            std::vector<int> (x.begin()+bps[i], x.begin()+bps[i+1]),
            std::vector<double> (y.begin()+bps[i], y.begin()+bps[i+1]),
            std::vector<double> (y_.begin()+bps[i], y_.begin()+bps[i+1]),
            fit_type
        );
        intercept_list[i] = std::get<0>(reg_results);
        new_coefs = std::get<1>(reg_results);
        coefs_list[i] = std::vector<double> (new_coefs.size(), 0);
        for (int j=0; j<coefs_list[i].size(); j++) {
            coefs_list[i][j] = new_coefs[j];
        }
        new_residuals = std::get<2>(reg_results);
        residuals.insert(residuals.end(), new_residuals.begin(), new_residuals.end());
        ssrs[i] = std::accumulate(new_residuals.begin(), new_residuals.end(), 0.0);
    }
    return Rcpp::List::create(
        Named("intercepts") = intercept_list,
        Named("coefs") = coefs_list,
        Named("ssr_vals") = ssrs,
        Named("residuals") = residuals
    );
};

//' Compute the sum squared residuals matrix
//' @param x x values
//' @param y yvalues
//' @param fit_type the type of equation to fit (linear, exponential, or quadratic)
//' @return sum squared residual matrix
// [[Rcpp::export]]
std::vector<std::vector<double>> precompute_ssr_mat(Rcpp::IntegerVector x, Rcpp::NumericVector y, std::string fit_type) {
    int n = x.size();
    Rcpp::NumericVector y_ = transform_y(y, fit_type);
    std::vector<std::vector<double>> ssr_mat_ (n);
    for (int i=0; i<n; i++) {
        ssr_mat_[i] = std::vector<double> (n, std::numeric_limits<int>::max());
    }
    for (int start_point=0; start_point<n; start_point++) {
        for (int end_point=start_point+1; end_point<n; end_point++) {
            std::vector<int> curr_x = std::vector<int> (x.begin()+start_point, x.begin()+end_point+1);
            std::vector<double> curr_y = std::vector<double> (y.begin()+start_point, y.begin()+end_point+1);
            std::vector<double> curr_y_ = std::vector<double> (y_.begin()+start_point, y_.begin()+end_point+1);
            std::tuple<double, std::vector<double>, std::vector<double>> reg_results = single_segment_regression(curr_x, curr_y, curr_y_, fit_type);
            ssr_mat_[start_point][end_point] = std::accumulate(std::get<2>(reg_results).begin(), std::get<2>(reg_results).end(), 0.0);
        }
    }
    return ssr_mat_;
};

//' Get the breakpoint locations for the given x and y values, and given parameters
//' @param x x values
//' @param y y values
//' @param n_bps number of breakpoints to estimate
//' @param fit_type type of equation to fit (linear, exponential, or quadratic)
//' @param min_len the minimum number of x values between breakpoints
//' @param ssr_mat the sum squared residuals matrix
//' @return a named list where bps is the breakpoint indices and ssr is a list of ssr values for each segment
// [[Rcpp::export]]
Rcpp::List get_breakpoints(Rcpp::IntegerVector x, Rcpp::NumericVector y, int n_bp, std::string fit_type, int min_len, std::vector<std::vector<double>> ssr_mat) {
    if (ssr_mat.empty()) {
        cout << "The SSR matrix is not properly set. Set ssr_mat or call precomputeSSRMat before calling get_breakpoints." << endl;
        return Rcpp::List::create();
    }
    else if (n_bp < 1) {
        cout << "Number of breakpoints must be at least 1." << endl;
        return Rcpp::List::create();
    }
    else if (min_len < 1) {
        cout << "Minimum length must be at least 1." << endl;
        return Rcpp::List::create();
    }
    else {
        std::tuple<std::vector<int>, double> result = run(n_bp, min_len, ssr_mat);
        return Rcpp::List::create(
            Named("bps") = std::get<0>(result),
            Named("ssr") = std::get<1>(result)
        );
    }
};

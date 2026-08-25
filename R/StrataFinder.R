library(ggplot2)
library(patchwork)
library(gridExtra)

StrataFinder <- setRefClass("StrataFinder",
    fields = list (
        x = "numeric",
        y = "numeric",
        fit_type = "character",
        min_len = "numeric",
        ssr_mat = "list",
        bps_idxs = "numeric",
        bps = "numeric"
    ),
    methods = list (
        #' Precompute the sum squared residuals matrix
        #' @return the sum squared residual matrix
        precomputeSSRMat = function() {
            ssr_mat <<- precompute_ssr_mat(x, y, fit_type)
            return (invisible(ssr_mat))
        },
        #' Find a specified number of strata
        #' @param n number of breakpoings
        #' @param quiet whether or not to print updates
        #' @return a named list with the breakpoints and their indices
        findNBreakpoints = function(n, quiet=FALSE) {
            # If the SSR matrix hasn't been computed yet, compute it
            if (is.null(ssr_mat)) {
                print("The SSR matrix is not properly set. Set ssr_mat or call precomputeSSRMat() before calling get_breakpoints. Remember to also recompute the SSR matrix when changing the fit type or minimum segment length.")
                return()
            }
            # Run get_breakpoints C++ code to get the breakpoints and save the results
            start_time <- Sys.time()
            results <- get_breakpoints(x, y, n, fit_type, min_len, ssr_mat)
            bps_idxs <<- results$bps+1
            bps <<- x[bps_idxs]
            end_time <- Sys.time()
            if (!quiet) {
                print(paste("Breakpoints:", toString(bps)))
                print(paste("Time taken:", round(difftime(end_time, start_time, units="secs")[[1]], 4), "seconds"))
            }
            return (invisible(list("bps"=list(bps), "bps_idxs"=list(bps_idxs))))
        },
        #' Find the number of breakpoints and their location that best fit the data
        #' @param min_bps the minimum number of breakpoints to try
        #' @param max_bps the maximum number of breakpoints to try
        #' @param test they type of statistical test to do to compare the number of breakpoints (t, f, both, or either)
        #' @param early_stop TRUE to exit once increasing the number of breakpoints does not lead to significantly different residuals. FALSE to run all possible breakpoint locations.
        #' @param quiet TRUE to not print any updates, FALSE to print updates
        #' @return a named list with the best number of breakpoints, their locations, and their indices
        findBreakpoints = function(min_bps, max_bps, test="t", early_stop=TRUE, quiet=FALSE) {
            print("1")
            # If the minimum length isn't set, exit
            if (is.null(min_len)){
                print("min_len is not set. Exiting.")
                return()
            }
            # If the SSR matrix hasn't been computed yet, compute it
            if (is.null(ssr_mat)) {
                print("The SSR matrix is not properly set. Set ssr_mat or call precomputeSSRMat() before calling get_breakpoints. Remember to also recompute the SSR matrix when changing the fit type or minimum segment length.")
                return()
            }
            if (!fit_type %in% c("flat", "linear", "exponential", "quadratic")) {
                print("Unknown fit type. Fit type should be flat, linear, exponential, or quadratic.")
                return()
            }
            start_time <- Sys.time()
            # The we want to test for 0 breakpoints, only run peicewise regression and set breakpoints to 0
            if (min_bps == 0) {
                curr_residuals <- piecewise_regression(x, y, numeric(0), fit_type)
                best_n <- 0
                best_bps <- as.numeric(c())
            }
            # Otherwise, do an initial run getting breakpoints and running piecewise regression with those breakpoints
            else {
                curr_results <- get_breakpoints(x, y, min_bps, fit_type, min_len, ssr_mat)
                curr_residuals <- piecewise_regression(x, y, curr_results$bps+1, fit_type)
                best_n <- min_bps
                best_bps <- curr_results$bps
            }
            # Loop up to the maximum number of breakpoints
            stop_flag <- FALSE
            for (n in min_bps+1:max_bps) {
                # Get the breakpoints and run piecewise regression
                next_results <- get_breakpoints(x, y, n, fit_type, min_len, ssr_mat)
                next_residuals <- piecewise_regression(x, y, next_results$bps+1, fit_type)
                # Get the p value based on the test type
                p <- compare_residuals(curr_residuals$residuals, next_residuals$residuals, test)
                if (!quiet) {
                    print(paste("Breakpoints for", n, "breakpoints =", toString(next_results$bps)))
                    print(paste("P-value between", n-1, "and", n, "=", p))
                }
                # Update residuals
                curr_residuals <- next_residuals
                # If these are better residuals, update the breakpoints
                if (!stop_flag & p < 0.05) {
                    best_n <- n
                    best_bps <- next_results$bps
                }
                # If the residuals aren't better and we are stopping early, return.
                else {
                    stop_flag <- TRUE
                    if (early_stop) {
                        break
                    }
                }
            }
            end_time <- Sys.time()
            if (!quiet) {
                print("------------------------------")
                print(paste("Best number of breakpoints:", best_n))
                print(paste("Breakpoints:", toString(best_bps)))
                print(paste("Time taken:", round(difftime(end_time, start_time, units="secs"), 4), "seconds"))
            }
            bps_idxs <<- best_bps
            bps <<- x[bps_idxs]
            return (invisible(list("n_bps"=best_n, "bps"=list(bps), "bps_idxs"=list(bps_idxs))))
        },
        #' Perform a statistical comparison between the two sets of residuals
        #' @param residuals1 the first set of residuals to compare
        #' @param residuals2 the second set of residuals to compare
        #' @param test the type of statistical test to perform ("t", "f", "either", or "both")
        #' @return the p-value of the statistical test (in the case of "either, this is the lower p-value of the two tests, in the case of "both", this is the higher p-value of the two tests)
        compare_residuals = function(residuals1, residuals2, test="t") {
            if (test == "t") {
                p <- t.test(abs(residuals1), abs(residuals2), var.equal=TRUE)$p.value
            } else if (test == "f") {
                p <- var.test(abs(residuals1), abs(residuals2))$p.value
            } else if (test == "either") {
                p1 <- t.test(abs(residuals1), abs(residuals2), var.equal=TRUE)$p.value
                p2 <- var.test(abs(residuals1), abs(residuals2))$p.value
                p <- min(p1, p2)
            } else if (test == "both") {
                p1 <- t.test(abs(residuals1), abs(residuals2), var.equal=TRUE)$p.value
                p2 <- var.test(abs(residuals1), abs(residuals2))$p.value
                p <- max(p1, p2)
            } else {
                print("Invalid test type. Test can be 't', 'f', 'either', or 'both'.")
                return (invisible(NULL))
            }
            return (invisible(p))
        },
        #' Get the error range of each breakpoint
        #' @param test statistical test type to compare residual distributions (t, f, either, or both)
        #' @return named list with the bounds and SSR values for each breakpoint
        get_bounds = function(test="t") {
            bounds <- data.frame()
            ssrs <- c()
            # Get results for actual breakpoint
            reg1 <- piecewise_regression(x, y, bps_idxs, fit_type)
            for (i in 1:length(bps_idxs)) {
                bp_ssrs <- c(sum(reg1$ssr_vals))
                # Find minimum location for this breakpoint such that the residuals of the new location 
                # and the residuals given the true location are not significantly different.
                tmp_bps <- bps_idxs
                # Get residuals for new breakpoint
                reg2 <- piecewise_regression(x, y, tmp_bps, fit_type)
                # While residuals aren't different, decrease the breakpoint location
                while (tmp_bps[i]>1 & sf$compare_residuals(reg1$residuals, reg2$residuals, test) > 0.05) {
                    bp_ssrs <- c(sum(reg2$ssr_vals), bp_ssrs)
                    tmp_bps[i] <- tmp_bps[i]-1
                    if (i > 1) {if (tmp_bps[i] < tmp_bps[i-1]) { break }}
                    reg2 <- piecewise_regression(x, y, tmp_bps, fit_type)
                }
                min_bp <- tmp_bps[i]
                # Find maximum location for this breakpoint such that the residuals of the new location 
                # and the residuals given the true location are not significantly different.                
                tmp_bps <- bps_idxs
                # Get residuals for new breakpoint
                reg2 <- piecewise_regression(x, y, tmp_bps, fit_type)
                # While residuals aren't different, increase the breakpoint location
                while (tmp_bps[i]<length(x)-1 & sf$compare_residuals(reg1$residuals, reg2$residuals, test) > 0.05) {
                    bp_ssrs <- c(bp_ssrs, sum(reg2$ssr_vals))
                    tmp_bps[i] <- tmp_bps[i]+1
                    if (i < length(bps)) {if (tmp_bps[i] > tmp_bps[i+1]) { break }}
                    reg2 <- piecewise_regression(x, y, tmp_bps, fit_type)
                }
                max_bp <- tmp_bps[i]
                min_bp_idx <- min_bp
                max_bp_idx <- max_bp
                min_bp <- x[min_bp_idx]
                max_bp <- x[max_bp_idx]
                bounds <- rbind(bounds, data.frame("bp"=bps[i], "bp_min"=min_bp, "bp_max"=max_bp, "bp_min_idx"=min_bp_idx, "bp_max_idx"=max_bp_idx))
                ssrs <- c(ssrs, list(bp_ssrs))
            }
            return (invisible(list("bounds"=bounds, "ssrs"=ssrs)))
        },
        #' Run piecewise regression over the data
        #' @return a named list of the intercept, coefficients, sum squared residuals, and residual values for each segment
        piecewiseRegression = function() {
            return (invisible(piecewise_regression(x, y, bps_idxs, fit_type)))
        },
        #' Run piecewise regression over the data
        #' @return a named list of the intercept, coefficients, sum squared residuals, and residual values for each segment
        piecewiseRegression = function(bps_idxs) {
            return (invisible(piecewise_regression(x, y, bps_idxs, fit_type)))
        },
        #' Plot the data with the current saved breakpoints and fit type
        #' @param show_error TRUE to show sum squared residual error, FALSE to just show breakpoints
        #' @return a ggplot2 plot object
        plot = function(show_error=FALSE) {
            # Set the fit function appropriately
            if (fit_type == "exponential") {
                fit_func <- function(x, intercept, coefs) exp(intercept + coefs[1]*x)
            }
            else if (fit_type == "quadratic") {
                fit_func <- function(x, intercept, coefs) intercept + coefs[1]*x + coefs[2]*x*x
            }
            else if (fit_type == "linear") {
                fit_func <- function(x, intercept, coefs) intercept + coefs[1]*x
            }
            else if (fit_type == "flat") {
                fit_func <- function(x, intercept, coefs) intercept
            }
            else {
                print("Unknown fit type. Fit type may be flat, linear, exponential, or quadratic.")
            }
            # Get the equations and residuals
            reg <- piecewise_regression(x, y, bps_idxs, fit_type)

            # If we have breakpoints and want to show error, calculate the bounds and the SSR values
            if (length(bps) > 0 & show_error) {
                bounds <- get_bounds()
                ssrs <- data.frame()
                for (i in 1:length(bps)) {
                    x_vals <- x[bounds$bounds[i,]$bp_min_idx:bounds$bounds[i,]$bp_max_idx]
                    ssrs <- rbind(ssrs, data.frame("x"=as.numeric(x_vals), "y"=as.numeric(bounds$ssrs[i][[1]])))
                }
            }

            # Plot the data points
            plt <- ggplot()
            plt <- plt + geom_point(aes(x=x, y=y), size=0.5) 
            plt <- plt + xlim(0, max(x)) + xlab("") + ylab("") 

            # If we have breakpoints, plot them
            if (length(bps) > 0) {
                # Plot the breakpoints
                plt <- plt + geom_vline(xintercept=bps, color="blue", linetype="dashed", linewidth=.75)
                
                # Plot the functions
                for (i in 1:length(bps)) {
                    if (i == 1) { xm = 0 } else {xm = bps[i-1]}
                        plt <- plt + geom_function(fun = fit_func, xlim=c(xm, bps[i]), args=c(reg$intercepts[i], reg$coefs[i]), color="blue", linewidth=0.5)
                }
                plt <- plt + geom_function(fun = fit_func, xlim=c(bps[length(bps)],x[length(x)]), args=c(reg$intercepts[length(reg$intercepts)], reg$coefs[length(reg$coefs)] ), color="blue", linewidth=0.5)
                
                if (show_error) {
                    # Add boundary boxes to the main plot
                    plt_ranges <- ggplot_build(plt)$layout$panel_params[[1]]$y.range
                    plt <- plt + geom_rect(aes(xmin=bounds$bounds$bp_min, xmax=bounds$bounds$bp_max, ymin=-Inf, ymax=Inf), alpha=0.25)

                    # Create SSR plot
                    plt2 <- ggplot() + xlim(0, max(x))
                    plt2 <- plt2 + geom_point(data=ssrs, aes(x=x, y=y), size=0.01)
                    plt2 <- plt2 + geom_vline(xintercept=bps, color="blue")
                    plt2 <- plt2 + xlab("") + ylab("")# + scale_y_continuous(labels="scientific")
                    plt2 <- plt2 + geom_rect(aes(xmin=bounds$bounds$bp_min, xmax=bounds$bounds$bp_max, ymin=-Inf, ymax=Inf), alpha=0.25)
                    plt <- plt + coord_cartesian(xlim=c(0,max(x))) 
                    plt2 <- plt2 + coord_cartesian(xlim=c(0,max(x)))
                    # plt <- grid.arrange(plt, plt2, nrow=2)
                    plt <- plt / plt2
                }        
            }
            # If we don't have any breakpoints, just plot the function
            else {
                plt <- plt + geom_function(fun = fit_func, xlim=c(0, x[length(x)]), args=c(reg$intercepts[1], reg$coefs[1]), color="blue")
            }
            return(plt)
        }
    )
)
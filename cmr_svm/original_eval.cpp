#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cassert>
#include <chrono>
#include <thread>
#include <svm.h> // libSVM headers

#include "cones.h"
#define NUM_THREADS 4

typedef std::vector<std::pair<double, double>> conesList;

/* takes a vector of points and the current point,
   returns the index of the closest point and the distance between that point and the current point */ 
std::pair<size_t, double> getClosestPointIdx(const conesList& points, const std::pair<double, double>& curr_point) {
    assert(!points.empty());

    size_t closest_idx = 0;
    double min_dist_squared = std::numeric_limits<double>::max();

    for (size_t i = 0; i < points.size(); ++i) {
        double dx = points[i].first - curr_point.first;
        double dy = points[i].second - curr_point.second;
        double dist_squared = dx * dx + dy * dy; 

        if (dist_squared < min_dist_squared) {
            min_dist_squared = dist_squared;
            closest_idx = i;
        }
    }

    // return the closest index and the square root of the minimum distance 
    return {closest_idx, std::sqrt(min_dist_squared)};
}

// takes a vector of points and returns the index of the spline starting index
size_t getSplineStartIdx(conesList& points) {
    // gets index of point with lowest y-axis value in points

    // first find minimum points
    double min_y = std::numeric_limits<double>::max();
    for (auto& point : points) {
        if (point.second < min_y) {
            min_y = point.second;
        }
    }

    // find points with y == min_y
    std::vector<size_t> idxs;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].second == min_y) {
            idxs.push_back(i);
        }
    }

    // take point closest to x = 0
    size_t closest_x_idx = idxs[0]; 
    double min_abs_x = std::abs(points[idxs[0]].first);
    for (size_t i = 1; i < idxs.size(); ++i) {
        double abs_x = std::abs(points[idxs[i]].first);
        if (abs_x < min_abs_x) {
            min_abs_x = abs_x;
            closest_x_idx = idxs[i];
        }
    }

    return closest_x_idx;
}

// takes a vector of points and sorts them based on a spline
conesList sortBoundaryPoints(conesList points, double max_spline_length=30) {
    // initialize spline length and sorted points
    double spline_length = 0;
    conesList sorted_points;

    // start from the lowest point along the y-axis
    size_t idx = getSplineStartIdx(points);
    std::pair<double, double> curr_point = points[idx];

    // remove the element at idx
    conesList rem_points = points;
    rem_points.erase(rem_points.begin() + idx);

    // add current point to sorted points
    sorted_points.push_back(curr_point);

    while (!rem_points.empty() && spline_length < max_spline_length) {

        // find closest point to curr_point
        double dist;
        std::pair<size_t, double> values = getClosestPointIdx(rem_points, curr_point);
        idx = values.first;
        dist = values.second;
        spline_length = spline_length + dist;

        // update iterates
        curr_point = rem_points[idx];
        rem_points.erase(rem_points.begin() + idx);

        // add closest point to sorted points
        sorted_points.push_back(curr_point);
    }

    return sorted_points;
}

// create a 2D mesh grid from a feature matrix (X[0] and X[1] are coordinates, X comes from conesToXY)
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> createMeshGrid(const std::vector<std::vector<double>>& X) {
    double x_min = X[0][0], x_max = X[0][0];
    double y_min = X[0][1], y_max = X[0][1];

    for (const auto& row : X) {
        x_min = std::min(x_min, row[0]);
        x_max = std::max(x_max, row[0]);
        y_min = std::min(y_min, row[1]);
        y_max = std::max(y_max, row[1]);
    }

    x_min -= 1.0;
    x_max += 1.0;
    y_min -= 1.0;
    y_max += 1.0;

    // ranges for x and y
    std::vector<double> x_range, y_range;
    for (double x = x_min; x <= x_max; x += 0.1) {
        x_range.push_back(x);
    }
    for (double y = y_min; y <= y_max; y += 0.1) {
        y_range.push_back(y);
    }

    // create the meshgrid
    std::vector<std::vector<double>> xx, yy;
    for (double y : y_range) {
        std::vector<double> x_row;
        std::vector<double> y_row;
        for (double x : x_range) {
            x_row.push_back(x);
            y_row.push_back(y);
        }
        xx.push_back(x_row);
        yy.push_back(y_row);
    }

    return {xx, yy};
}

// flatten the mesh grid into a vector of points, XY is output of createMeshGrid
std::vector<std::vector<double>> flattenMesh(const std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>& XY) {
    std::vector<std::vector<double>> xx = XY.first;
    std::vector<std::vector<double>> yy = XY.second;

    // flatten xx, yy
    std::vector<std::vector<double>> svm_input;
    for (size_t i = 0; i < xx.size(); ++i) {
        for (size_t j = 0; j < xx[i].size(); ++j) {
            svm_input.push_back({xx[i][j], yy[i][j]});
        }
    }

    return svm_input;
}


/* reshape the svm_output (array) into a 2D grid of points, svm_output comes from SVM prediction
   and xx is first part of XY pair, XY comes from createMeshGrid */ 
std::vector<std::vector<double>> reshapeOutput(const std::vector<double>& svm_output, std::vector<std::vector<double>> xx) {
    std::vector<std::vector<double>> Z;
    size_t rows = xx.size();
    size_t cols = xx[0].size();
    for (size_t i = 0; i < rows; ++i) {
        Z.emplace_back(svm_output.begin() + i * cols, svm_output.begin() + (i + 1) * cols);
    }

    return Z;
}

// predict the value of a node
double nodePredictor(const std::vector<double>& cone, const svm_model* model) {
    svm_node* node = new svm_node[cone.size() + 1];
    for (size_t i = 0; i < cone.size(); ++i) {
        node[i].index = i + 1;
        node[i].value = cone[i];
    }
    node[cone.size()].index = -1; 
    double value = svm_predict(model, node);
    delete[] node;
    return value;
}

/* take the flatten mesh and generate a vector of boundary points */
conesList boundaryDetection(const std::vector<std::vector<double>>& Z, const std::vector<std::vector<double>>& xx, 
                            const std::vector<std::vector<double>>& yy) {
    size_t rows = xx.size();
    size_t cols = xx[0].size();
    
    conesList boundary_points;
    
    // extract the top left (Z_TL), bottom right (Z_BR), and central region (Z_C)
    for (size_t i = 0; i < rows - 1; ++i) {
        for (size_t j = 0; j < cols - 1; ++j) {
            double Z_TL = Z[i][j];
            double Z_BR = Z[i+1][j];
            double Z_C  = Z[i+1][j+1];

            // if Z_C has a different label than Z_TL or Z_BR, it is a boundary point
            if (Z_C != Z_TL || Z_C != Z_BR) {
                boundary_points.emplace_back(xx[i+1][j+1], yy[i+1][j+1]);
            }
        }
    }

    return boundary_points;
}

// downsample the boundary points 
conesList downsamplePoints(const conesList& boundary_points) {
    conesList downsampled;
    double accumulated_dist = 0.0;

    for (size_t i = 1; i < boundary_points.size(); ++i) {
        auto& p0 = boundary_points[i - 1];
        auto& p1 = boundary_points[i];
        double dist = std::sqrt(std::pow(p1.first - p0.first, 2) + std::pow(p1.second - p0.second, 2));
        accumulated_dist += dist;

        if (std::abs(accumulated_dist - 0.5) < 0.1) {
            downsampled.push_back(p1);
            accumulated_dist = 0.0;
        }

        if (accumulated_dist > 0.55) {
            accumulated_dist = 0.0;
        }
    }

    return downsampled;
}


conesList cones_to_midline(Cones cones) {
    // check if there are no blue or yellow cones
    const auto& blue_cones = cones.getBlueCones();
    const auto& yellow_cones = cones.getYellowCones();

    if (blue_cones.empty() && yellow_cones.empty()) {
        return conesList(); 
    }

    // augment dataset to make it better for SVM training
    cones.supplementCones();
    cones = cones.augmentConesCircle(cones, 10, 1.2);
    std::cout << "blue cones size after augmenting: " << cones.getBlueCones().size() << std::endl;
    std::cout << "yellow cones size after augmenting: " << cones.getYellowCones().size() << std::endl;

    // acquire the feature matrix and label vector
    std::pair<std::vector<std::vector<double>>, std::vector<double>> xy = cones.conesToXY(cones);
    std::vector<std::vector<double>> X = xy.first;
    std::vector<double> y = xy.second;

    // prepare SVM data
    svm_problem prob;
    prob.l = X.size(); // number of training examples
    prob.y = new double[prob.l]; // labels
    prob.x = new svm_node*[prob.l]; // feature vectors

    for (int i = 0; i < prob.l; ++i) {
        prob.y[i] = y[i]; // set the label for each example

        // create the feature vector
        prob.x[i] = new svm_node[X[i].size() + 1]; // +1 for the end marker
        for (size_t j = 0; j < X[i].size(); ++j) {
            prob.x[i][j].index = j + 1; // 1-based indexing for libSVM
            prob.x[i][j].value = X[i][j];
        }
        prob.x[i][X[i].size()].index = -1; // End marker
    }

    /* this section of code used to SVM parameters to match scikit-learn parameters */
    // first, compute mean of all elements in X 
    double sum_all = 0.0;

    int N = static_cast<int>(X.size());
    if (N == 0 || X[0].empty()) {
        std::cerr << "No training data available for SVM.\n";
        return conesList();
    }
    int d = static_cast<int>(X[0].size());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            sum_all += X[i][j];
        }
    }
    double mean_all = sum_all / (N * d);

    // then, compute variance over all elements in X
    double sum_var = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            double diff = X[i][j] - mean_all;
            sum_var += diff * diff;
        }
    }
    double var_all = sum_var / (N * d);

    // calculate gamma_scale to match scikit-learn
    double gamma_scale = 1.0 / (d * var_all);

    // set up svm
    svm_parameter param;
    memset(&param, 0, sizeof(param));
    param.svm_type = C_SVC;
    param.kernel_type = POLY;
    param.degree = 3;     
    param.C = 10.0;         
    param.coef0 = 1.0;     
    param.gamma = gamma_scale;
    param.cache_size = 200; 
    param.eps = 0.001;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    const char *error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
        std::cerr << "Error in SVM parameters: " << error_msg << std::endl;
        return std::vector<std::pair<double, double>>(); 
    }

    // train the SVM model
    svm_model* model = svm_train(&prob, &param);

    // create meshgrid
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> XY = createMeshGrid(X);
    std::vector<std::vector<double>> xx = XY.first;
    std::vector<std::vector<double>> yy = XY.second;

    // flatten xx, yy 
    std::vector<std::vector<double>> svm_input = flattenMesh(XY);

    
    // predict using the SVM Model
    std::vector<double> svm_output;
    for (const std::vector<double>& input : svm_input) {
        svm_node* node = new svm_node[input.size() + 1];
        for (size_t i = 0; i < input.size(); ++i) {
            node[i].index = i + 1;
            node[i].value = input[i];
        }
        node[input.size()].index = -1; 
        svm_output.push_back(svm_predict(model, node));
        delete[] node;
    }


    // reshape Z to match the shape of xx and yy
    std::vector<std::vector<double>> Z = reshapeOutput(svm_output, xx);

    // boundary detection
    conesList boundary_points = boundaryDetection(Z, xx, yy);
    std::cout << "boundary points size after detecting: " << boundary_points.size() << std::endl;
    std::cout << "before sorting: " << std::endl;

    for (const auto& p : boundary_points) {
        std::cout << "(" << p.first << ", " << p.second << ")," << std::endl;
    }

    // sort boundary points 
    if (boundary_points.empty()) {
        std::cerr << "No boundary points found. Returning empty vector.\n";
        return conesList();
    }

    boundary_points = sortBoundaryPoints(boundary_points);
    std::cout << "boundary points size after sorting: " << boundary_points.size() << std::endl;
    std::cout << "before downsampling: " << std::endl;

    for (const auto& p : boundary_points) {
        std::cout << "(" << p.first << ", " << p.second << ")," << std::endl;
    }

    // downsample boundary points
    conesList downsampled = downsamplePoints(boundary_points);
    std::cout << "boundary points size after downsampling: " << downsampled.size() << std::endl;

    // free allocated memory
    for (int i = 0; i < prob.l; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;

    svm_free_and_destroy_model(&model);

    return downsampled;
}

double ator(int a){
    return (double) a * M_PI / 180.0;
}

// feet to m
double ftom(int a){
    return (double) a * 0.3048;
}

// testing cones to midline on squidward track
int main() {
    std::vector<std::vector<double>> blue_list = {
        {-4, 0},
        {-4, 2},
        {-4, 4},
        {-4, 6},
        {-4, 8},
        {-4, 10},
        {-4, 12},
        {-4, 14},
        {-4, 16},
        {-4, 18},
        {-4, 20},
        {-4, 22},
        {-4 - 2 + 2 * cos(ator(30)), 22 + 2 * sin(ator(30))},
        {-4 - 2 + 2 * cos(ator(60)), 22 + 2 * sin(ator(60))},
        {-6, 24},
        {-8, 24},
        {-10, 24},
        {-12, 24},
        {-14, 24},
        {-14 + 2 * cos(ator(120)), 24 - 2 + 2 * sin(ator(120))},
        {-14 + 2 * cos(ator(150)), 24 - 2 + 2 * sin(ator(150))},
        {-16, 22},
        {-16 + ftom(4), 20},
        {-16 + ftom(6), 18},
        {-16 + ftom(2), 16},
        {-16 - ftom(2), 14},
        {-16 - ftom(6), 12},
        {-16 - ftom(4), 10},
        {-16, 8},
        {-16, 6},
        {-16, 4},
        {-16 + 2 - 2 * cos(ator(30)), 4 - 2 * sin(ator(30))},
        {-16 + 2 - 2 * cos(ator(60)), 4 - 2 * sin(ator(90))},
        {-14, 2},
        {-12, 2},
        {-10, 2},
        {-8, 2},
        {-6, 2},
        {-4, 2}
    };


    std::vector<std::vector<double>> yellow_list = {
        {0, 0},
        {0, 2},
        {0, 4},
        {0, 6},
        {0, 8},
        {0, 10},
        {0, 12},
        {0, 14},
        {0, 16},
        {0, 18},
        {0, 20},
        {0, 22},
        {0 - 6 + 6 * cos(ator(30)), 22 + 6 * sin(ator(30))},
        {0 - 6 + 6 * cos(ator(60)), 22 + 6 * sin(ator(60))},
        {-6, 28},
        {-8, 28},
        {-10, 28},
        {-12, 28},
        {-14, 28},
        {-14 + 6 * cos(ator(120)), 28 - 6 + 6 * sin(ator(120))},
        {-14 + 6 * cos(ator(150)), 28 - 6 + 6 * sin(ator(150))},
        {-20, 22},
        {-20 + ftom(4), 20},
        {-20 + ftom(6), 18},
        {-20 + ftom(2), 16},
        {-20 - ftom(2), 14},
        {-20 - ftom(6), 12},
        {-20 - ftom(4), 10},
        {-20, 8},
        {-20, 6},
        {-20, 4},
        {-16 + 2 - 6 * cos(ator(30)), 4 - 6 * sin(ator(30))},
        {-16 + 2 - 6 * cos(ator(60)), 4 - 6 * sin(ator(90))},
        {-14, -2},
        {-12, -2},
        {-10, -2},
        {-8, -2},
        {-6, -2},
        {-4, -2}
    };

    std::vector<std::vector<double>> blue_straight = {
        {-4, 0},
        {-4, 2},
        {-4, 4},
        {-4, 6},
        {-4, 8},
        {-4, 10},
        {-4, 12},
        {-4, 14},
        {-4, 16},
        {-4, 18},
        {-4, 20},
        {-4, 22}
    };


    std::vector<std::vector<double>> yellow_straight = {
        {0, 0},
        {0, 2},
        {0, 4},
        {0, 6},
        {0, 8},
        {0, 10},
        {0, 12},
        {0, 14},
        {0, 16},
        {0, 18},
        {0, 20},
        {0, 22}
    };

    Cones conesList3;
    conesList3.addMultipleBlue(blue_straight);
    conesList3.addMultipleYellow(yellow_straight);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<double, double>> overallAns2 = cones_to_midline(conesList3);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    std::cout << "straight points" << std::endl;

    for (const auto& p : overallAns2) {
        std::cout << "(" << p.first << ", " << p.second << ")," << std::endl;
    }
    /*
    Cones conesList2;
    conesList2.addMultipleBlue(blue_list);
    conesList2.addMultipleYellow(yellow_list);
    std::vector<std::pair<double, double>> overallAns = cones_to_midline(conesList2);

    std::cout << "overall midline: " << std::endl;

    for (const auto& p : overallAns) {
        std::cout << "(" << p.first << ", " << p.second << ")," << std::endl;
    }
    */

    return 0;
}
#include <iostream>
#include <vector>
#include <cmath>

double calculate_std_dev(std::vector<double> numbers) {
    double sum = 0.0;
    for (double n : numbers) {
        sum += n;
    }
    double mean = sum / numbers.size();
    double variance_sum = 0.0;
    for (double n : numbers) {
        variance_sum += (n - mean) * (n - mean);
    }
    double variance = variance_sum / numbers.size();
    return std::sqrt(variance);
}

int main() {
    std::vector<double> my_numbers = {10.0, 12.0, 23.0, 23.0, 16.0};
    double result = calculate_std_dev(my_numbers);
    std::cout << "Standard Deviation:" << result << std::endl;
    return 0;
}


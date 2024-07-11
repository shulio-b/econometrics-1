import scipy.stats as stats

class Econometrics:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_mean = sum(x) / len(x)
        self.y_mean = sum(y) / len(y)
        self.beta1 = self.calculate_beta1()
        self.beta0 = self.calculate_beta0()
        self.y_hat = self.calculate_y_hat()
        self.sse = self.calculate_sse()
        self.sst = self.calculate_sst()
        self.ssr = self.calculate_ssr()

    def import_table(self):
        return list(zip(self.x, self.y))

    def calculate_beta1(self):
        numerator = sum((xi - self.x_mean) * (yi - self.y_mean) for xi, yi in zip(self.x, self.y))
        denominator = sum((xi - self.x_mean) ** 2 for xi in self.x)
        return numerator / denominator

    def calculate_beta0(self):
        return self.y_mean - self.beta1 * self.x_mean

    def xi_minus_average_x(self):
        return [xi - self.x_mean for xi in self.x]

    def sum_xi_minus_average_x(self):
        return sum((xi - self.x_mean) for xi in self.x)
    
    def square_sum_xi_minus_average_x(self):
        return sum((xi - self.x_mean) ** 2 for xi in self.x)

    def yi_minus_average_y(self):
        return [yi - self.y_mean for yi in self.y]

    def xi_minus_average_x_squared(self):
        return [(xi - self.x_mean) ** 2 for xi in self.x]

    def variance(self):
        return sum(self.xi_minus_average_x_squared()) / (len(self.x) - 1)

    def calculate_y_hat(self):
        return [self.beta0 + self.beta1 * xi for xi in self.x]

    def calculate_sse(self):
        return sum((yi - y_hat) ** 2 for yi, y_hat in zip(self.y, self.y_hat))

    def calculate_sst(self):
        return sum((yi - self.y_mean) ** 2 for yi in self.y)

    def calculate_ssr(self):
        return self.sst - self.sse

    def r_squared(self):
        return 1 - (self.sse / self.sst)

    def variance_beta1(self):
        return self.sse / self.square_sum_xi_minus_average_x()

    def critical_value(self, alpha=0.05):
        df = len(self.x) - 2  # degrees of freedom
        return stats.t.ppf(1 - alpha / 2, df)

# Updated arrays:
x = [59, 71, 70, 64, 72, 73, 64, 70, 68, 75]
y = [165, 167, 175, 179, 170, 169, 164, 173, 174, 175]

model = Econometrics(x, y)
print("Imported table:", model.import_table())
print("Beta0:", model.beta0)
print("Beta1:", model.beta1)
print("y^ (predicted y):", model.y_hat)
print("SSE (Sum of Squared Errors):", model.sse)
print("SST (Total Sum of Squares):", model.sst)
print("SSR (Regression Sum of Squares):", model.ssr)
print("R² (Coefficient of Determination):", model.r_squared())
print("Sum of (xi - mean x):", model.sum_xi_minus_average_x())
print("Square sum of (xi - mean x)", model.square_sum_xi_minus_average_x())
print("Variance of Beta1:", model.variance_beta1())
print("Critical value (t-distribution):", model.critical_value())
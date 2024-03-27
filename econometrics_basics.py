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
        self.sqe = self.calculate_sqe()
        self.sqt = self.calculate_sqt()

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

    def yi_minus_average_y(self):
        return [yi - self.y_mean for yi in self.y]

    def xi_minus_average_x_squared(self):
        return [(xi - self.x_mean) ** 2 for xi in self.x]

    def variance(self):
        return sum(self.xi_minus_average_x_squared()) / (len(self.x) - 1)

    def calculate_y_hat(self):
        return [self.beta0 + self.beta1 * xi for xi in self.x]

    def calculate_sqe(self):
        return sum((yi - y_hat) ** 2 for yi, y_hat in zip(self.y, self.y_hat))

    def calculate_sqt(self):
        return sum((yi - self.y_mean) ** 2 for yi in self.y)

    def r_squared(self):
        return 1 - (self.sqe / self.sqt)

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
print("SQE (Sum of Squared Errors):", model.sqe)
print("SQT (Total Sum of Squares):", model.sqt)
print("R² (Coefficient of Determination):", model.r_squared())
print("Critical value (t-distribution):", model.critical_value())

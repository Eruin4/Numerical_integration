import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time
import os

# 한글 폰트 설정 (matplotlib에서 한글 표시를 위해)
import matplotlib.font_manager as fm
# Windows에서 한글 폰트 설정
try:
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/gulim.ttc'  # 굴림
    if not os.path.exists(font_path):
        plt.rcParams['font.family'] = 'DejaVu Sans'
    else:
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
plt.rcParams['axes.unicode_minus'] = False

class NumericalIntegrationAnalyzer:
    def __init__(self):
        # 분석할 함수들과 정확한 해석적 적분값
        self.functions = {
            'x^3': {
                'func': lambda x: x**3,
                'exact': 0.25,  # [x^4/4] from 0 to 1
                'name': 'x³'
            },
            'ln(1+x)': {
                'func': lambda x: np.log(1 + x),
                'exact': 2*np.log(2) - 1,  # 2ln(2) - 1
                'name': 'ln(1+x)'
            },
            'sin(50x)': {
                'func': lambda x: np.sin(50 * x),
                'exact': (1 - np.cos(50))/50,  # [-cos(50x)/50] from 0 to 1
                'name': 'sin(50x)'
            },
            'exp(-x^2)': {
                'func': lambda x: np.exp(-x**2),
                'exact': integrate.quad(lambda x: np.exp(-x**2), 0, 1)[0],  # 수치적으로 계산
                'name': 'e^(-x²)'
            }
        }
    
    def trapezoidal_rule(self, func, a, b, n):
        """사다리꼴 공식"""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    def simpson_rule(self, func, a, b, n):
        """심슨 공식 (n은 짝수여야 함)"""
        if n % 2 != 0:
            n += 1  # 홀수면 짝수로 만듦
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        
        # 심슨 공식: h/3 * [f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + f(xn)]
        result = y[0] + y[-1]
        for i in range(1, n, 2):
            result += 4 * y[i]
        for i in range(2, n-1, 2):
            result += 2 * y[i]
        
        return h * result / 3
    
    def monte_carlo_integration(self, func, a, b, n):
        """몬테카를로 방법"""
        np.random.seed(42)  # 재현 가능한 결과를 위해
        x_random = np.random.uniform(a, b, n)
        y_values = func(x_random)
        return (b - a) * np.mean(y_values)
    
    def analyze_convergence(self, func_name, max_points=10000):
        """주어진 함수에 대해 세 방법의 수렴성 분석"""
        func_info = self.functions[func_name]
        func = func_info['func']
        exact_value = func_info['exact']
        
        # 다양한 분할 수
        n_values = np.logspace(1, np.log10(max_points), 50).astype(int)
        
        errors_trap = []
        errors_simp = []
        errors_mc = []
        times_trap = []
        times_simp = []
        times_mc = []
        
        for n in n_values:
            # 사다리꼴 방법
            start_time = time.time()
            trap_result = self.trapezoidal_rule(func, 0, 1, n)
            times_trap.append(time.time() - start_time)
            errors_trap.append(abs(trap_result - exact_value))
            
            # 심슨 방법
            start_time = time.time()
            simp_result = self.simpson_rule(func, 0, 1, n)
            times_simp.append(time.time() - start_time)
            errors_simp.append(abs(simp_result - exact_value))
            
            # 몬테카를로 방법
            start_time = time.time()
            mc_result = self.monte_carlo_integration(func, 0, 1, n)
            times_mc.append(time.time() - start_time)
            errors_mc.append(abs(mc_result - exact_value))
        
        return {
            'n_values': n_values,
            'errors_trap': np.array(errors_trap),
            'errors_simp': np.array(errors_simp),
            'errors_mc': np.array(errors_mc),
            'times_trap': np.array(times_trap),
            'times_simp': np.array(times_simp),
            'times_mc': np.array(times_mc),
            'exact_value': exact_value
        }
    
    def plot_convergence_analysis(self, results, func_name):
        """수렴성 분석 결과 시각화"""
        func_display_name = self.functions[func_name]['name']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Numerical Integration Convergence Analysis: {func_display_name}', fontsize=16)
        
        # 1. 오차 vs 분할 수 (로그-로그 스케일)
        ax1.loglog(results['n_values'], results['errors_trap'], 'b-o', label='Trapezoidal Rule', markersize=3)
        ax1.loglog(results['n_values'], results['errors_simp'], 'r-s', label='Simpson Rule', markersize=3)
        ax1.loglog(results['n_values'], results['errors_mc'], 'g-^', label='Monte Carlo', markersize=3)
        ax1.set_xlabel('Number of subdivisions (n)')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Error Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 오차 vs 계산 시간
        ax2.loglog(results['times_trap'], results['errors_trap'], 'b-o', label='Trapezoidal Rule', markersize=3)
        ax2.loglog(results['times_simp'], results['errors_simp'], 'r-s', label='Simpson Rule', markersize=3)
        ax2.loglog(results['times_mc'], results['errors_mc'], 'g-^', label='Monte Carlo', markersize=3)
        ax2.set_xlabel('Computation Time (seconds)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Efficiency Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 수렴 차수 분석 (이론적 기울기와 비교)
        # 사다리꼴: O(h^2) = O(1/n^2)
        # 심슨: O(h^4) = O(1/n^4)  
        # 몬테카를로: O(1/sqrt(n))
        theoretical_trap = results['errors_trap'][0] * (results['n_values'][0] / results['n_values'])**2
        theoretical_simp = results['errors_simp'][0] * (results['n_values'][0] / results['n_values'])**4
        theoretical_mc = results['errors_mc'][0] * np.sqrt(results['n_values'][0] / results['n_values'])
        
        ax3.loglog(results['n_values'], results['errors_trap'], 'b-o', label='Trapezoidal (Actual)', markersize=3)
        ax3.loglog(results['n_values'], theoretical_trap, 'b--', label='Trapezoidal O(1/n²)', alpha=0.7)
        ax3.loglog(results['n_values'], results['errors_simp'], 'r-s', label='Simpson (Actual)', markersize=3)
        ax3.loglog(results['n_values'], theoretical_simp, 'r--', label='Simpson O(1/n⁴)', alpha=0.7)
        ax3.loglog(results['n_values'], results['errors_mc'], 'g-^', label='Monte Carlo (Actual)', markersize=3)
        ax3.loglog(results['n_values'], theoretical_mc, 'g--', label='Monte Carlo O(1/√n)', alpha=0.7)
        ax3.set_xlabel('Number of subdivisions (n)')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Theoretical vs Actual Convergence')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 계산 시간 비교
        ax4.loglog(results['n_values'], results['times_trap'], 'b-o', label='Trapezoidal Rule', markersize=3)
        ax4.loglog(results['n_values'], results['times_simp'], 'r-s', label='Simpson Rule', markersize=3)
        ax4.loglog(results['n_values'], results['times_mc'], 'g-^', label='Monte Carlo', markersize=3)
        ax4.set_xlabel('Number of subdivisions (n)')
        ax4.set_ylabel('Computation Time (seconds)')
        ax4.set_title('Computation Time Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'convergence_analysis_{func_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, all_results):
        """전체 분석 결과 요약 보고서 생성"""
        report = "=" * 80 + "\n"
        report += "수치적분 방법 수렴성 분석 보고서\n"
        report += "=" * 80 + "\n\n"
        
        for func_name, results in all_results.items():
            func_display_name = self.functions[func_name]['name']
            exact_value = results['exact_value']
            
            report += f"함수: {func_display_name}\n"
            report += f"정확한 값: {exact_value:.10f}\n"
            report += "-" * 50 + "\n"
            
            # 최종 오차 (가장 많은 분할 수에서)
            final_n = results['n_values'][-1]
            final_error_trap = results['errors_trap'][-1]
            final_error_simp = results['errors_simp'][-1]
            final_error_mc = results['errors_mc'][-1]
            
            report += f"분할 수 n = {final_n}에서의 결과:\n"
            report += f"  사다리꼴 방법: 오차 = {final_error_trap:.2e}\n"
            report += f"  심슨 방법:     오차 = {final_error_simp:.2e}\n"
            report += f"  몬테카를로:    오차 = {final_error_mc:.2e}\n"
            
            # 수렴 차수 추정
            mid_idx = len(results['n_values']) // 2
            end_idx = -1
            
            # 사다리꼴 수렴 차수
            trap_slope = -np.log(results['errors_trap'][end_idx] / results['errors_trap'][mid_idx]) / \
                        np.log(results['n_values'][end_idx] / results['n_values'][mid_idx])
            
            # 심슨 수렴 차수
            simp_slope = -np.log(results['errors_simp'][end_idx] / results['errors_simp'][mid_idx]) / \
                        np.log(results['n_values'][end_idx] / results['n_values'][mid_idx])
            
            # 몬테카를로 수렴 차수
            mc_slope = -np.log(results['errors_mc'][end_idx] / results['errors_mc'][mid_idx]) / \
                      np.log(results['n_values'][end_idx] / results['n_values'][mid_idx])
            
            report += f"\n관측된 수렴 차수:\n"
            report += f"  사다리꼴 방법: {trap_slope:.2f} (이론값: 2.0)\n"
            report += f"  심슨 방법:     {simp_slope:.2f} (이론값: 4.0)\n"
            report += f"  몬테카를로:    {mc_slope:.2f} (이론값: 0.5)\n"
            
            report += "\n" + "=" * 80 + "\n\n"
        
        with open('convergence_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report


def main():
    """메인 분석 함수"""
    analyzer = NumericalIntegrationAnalyzer()
    
    print("수치적분 방법의 수렴성 분석을 시작합니다...")
    print("분석 대상 함수들:")
    for func_name, func_info in analyzer.functions.items():
        print(f"  - {func_info['name']} (정확한 값: {func_info['exact']:.6f})")
    
    all_results = {}
    
    # 각 함수에 대해 분석 수행
    for func_name in analyzer.functions.keys():
        print(f"\n{analyzer.functions[func_name]['name']} 분석 중...")
        results = analyzer.analyze_convergence(func_name)
        all_results[func_name] = results
        analyzer.plot_convergence_analysis(results, func_name)
    
    # 요약 보고서 생성
    print("\n요약 보고서 생성 중...")
    report = analyzer.generate_summary_report(all_results)
    print("\n" + report)
    
    print("\n분석 완료! 결과는 'integral' 폴더에 저장되었습니다.")
    print("- 각 함수별 수렴성 그래프: convergence_analysis_*.png")
    print("- 요약 보고서: convergence_analysis_report.txt")


if __name__ == "__main__":
    main() 
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

plt.rcParams['axes.unicode_minus'] = False

class PseudoRandomAnalyzer:
    def __init__(self):
        # 여러 종류의 pseudo random 함수들
        self.functions = {
            'chaotic_map': {
                'func': self._chaotic_map_function,
                'exact': None,
                'name': 'Chaotic Map Function',
                'description': '카오스 맵 기반 함수'
            },
            'hash_based': {
                'func': self._hash_based_function,
                'exact': None,
                'name': 'Hash-based Function',
                'description': '해시 기반 의사 랜덤'
            },
            'noise_dominated': {
                'func': self._noise_dominated_function,
                'exact': None,
                'name': 'Noise-dominated Function',
                'description': '노이즈가 지배적인 함수'
            }
        }
        
        self._calculate_exact_values()
    
    def _chaotic_map_function(self, x):
        """카오스 맵 기반 의사 랜덤 함수 (벡터화 지원)"""
        # 로지스틱 맵을 이용한 카오스 시퀀스
        if np.isscalar(x):
            x = np.array([x])
            scalar_input = True
        else:
            scalar_input = False
            
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            # 로지스틱 맵: x_{n+1} = r * x_n * (1 - x_n)
            r = 3.99  # 카오스 매개변수
            seed = (xi * 1000) % 1  # 0~1 범위로 정규화
            
            # 여러 번 반복하여 카오스적 행동 생성
            for _ in range(10):
                seed = r * seed * (1 - seed)
            
            # 기본 함수에 카오스 노이즈 추가
            base = np.sin(2 * np.pi * xi)
            noise = 0.5 * (seed - 0.5)  # -0.25 ~ 0.25 범위
            result[i] = base + noise
        
        return result[0] if scalar_input else result
    
    def _hash_based_function(self, x):
        """해시 기반 의사 랜덤 함수 (벡터화 지원)"""
        if np.isscalar(x):
            x = np.array([x])
            scalar_input = True
        else:
            scalar_input = False
            
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            # 간단한 해시 함수
            seed = int((xi * 999983) % 1000000)  # 큰 소수 사용
            
            # 선형 합동 생성기 (LCG)
            a, c, m = 1664525, 1013904223, 2**32
            hash_val = ((a * seed + c) % m) / m
            
            # 기본 함수에 해시 기반 노이즈 추가
            base = 0.5 * np.sin(xi) + 0.3 * np.cos(3 * xi)
            noise = 0.4 * (hash_val - 0.5)
            result[i] = base + noise
        
        return result[0] if scalar_input else result
    
    def _noise_dominated_function(self, x):
        """노이즈가 지배적인 함수 (벡터화 지원)"""
        if np.isscalar(x):
            x = np.array([x])
            scalar_input = True
        else:
            scalar_input = False
            
        result = np.zeros_like(x)
        for i, xi in enumerate(x):
            # 여러 주파수의 의사 랜덤 신호 합성
            base_signal = 0.1 * np.sin(xi)
            
            # 의사 랜덤 노이즈 (다양한 주파수)
            noise = 0
            frequencies = [37, 73, 149, 293, 587]  # 소수들
            for j, freq in enumerate(frequencies):
                phase = (xi * freq * 1000) % (2 * np.pi)
                amplitude = 0.2 / (j + 1)
                noise += amplitude * np.sin(phase)
            
            result[i] = base_signal + noise
        
        return result[0] if scalar_input else result
    
    def _calculate_exact_values(self):
        """정확한 적분값들을 고정밀 수치적분으로 계산"""
        print("Pseudo random 함수들의 정확한 적분값 계산 중...")
        for func_name, func_info in self.functions.items():
            try:
                # 적응적 적분법으로 계산
                result, error = integrate.quad(func_info['func'], 0, 1, limit=500, epsabs=1e-10, epsrel=1e-10)
                func_info['exact'] = result
                print(f"  {func_info['name']}: {result:.10f} (오차: {error:.2e})")
            except Exception as e:
                print(f"  {func_info['name']}: 고정밀 적분 실패 - {e}")
                # 더 안정적인 방법으로 계산
                try:
                    func_info['exact'] = self._stable_integration(func_info['func'])
                    print(f"  {func_info['name']}: {func_info['exact']:.10f} (안정적 방법)")
                except:
                    print(f"  {func_info['name']}: 계산 실패")
                    func_info['exact'] = None
    
    def _stable_integration(self, func, n=100000):
        """안정적인 적분 계산"""
        x = np.linspace(0, 1, n+1)
        y = func(x)
        return np.trapz(y, x)
    
    def trapezoidal_rule(self, func, a, b, n):
        """사다리꼴 공식"""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    def simpson_rule(self, func, a, b, n):
        """심슨 공식"""
        if n % 2 != 0:
            n += 1
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = func(x)
        
        result = y[0] + y[-1]
        for i in range(1, n, 2):
            result += 4 * y[i]
        for i in range(2, n-1, 2):
            result += 2 * y[i]
        
        return h * result / 3
    
    def monte_carlo_integration(self, func, a, b, n):
        """몬테카를로 방법 (여러 번 실행하여 안정성 향상)"""
        results = []
        for seed in range(10):  # 10번 실행
            np.random.seed(42 + seed)
            x_random = np.random.uniform(a, b, n)
            y_values = func(x_random)
            results.append((b - a) * np.mean(y_values))
        return np.mean(results)
    
    def analyze_convergence(self, func_name, max_points=5000):
        """수렴성 분석"""
        func_info = self.functions[func_name]
        func = func_info['func']
        exact_value = func_info['exact']
        
        if exact_value is None:
            print(f"Warning: {func_name}의 정확한 값이 없습니다.")
            return None
        
        # 다양한 분할 수
        n_values = np.logspace(1, np.log10(max_points), 40).astype(int)
        
        errors_trap = []
        errors_simp = []
        errors_mc = []
        times_trap = []
        times_simp = []
        times_mc = []
        
        print(f"\n{func_info['name']} 분석 중...")
        print(f"설명: {func_info['description']}")
        print(f"정확한 값: {exact_value:.10f}")
        
        for i, n in enumerate(n_values):
            try:
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
                
                # 중간 결과 출력
                if i % 8 == 0:
                    print(f"  n={n:5d}: Trap={errors_trap[-1]:.3e}, Simpson={errors_simp[-1]:.3e}, MC={errors_mc[-1]:.3e}")
                    
            except Exception as e:
                print(f"Error at n={n}: {e}")
                # 오류시 건너뛰기
                continue
        
        return {
            'n_values': np.array(n_values[:len(errors_trap)]),
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
        if results is None:
            return
            
        func_info = self.functions[func_name]
        func_display_name = func_info['name']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Pseudo Random Function Analysis: {func_display_name}', fontsize=16)
        
        # 1. 오차 vs 분할 수
        ax1.loglog(results['n_values'], results['errors_trap'], 'b-o', label='Trapezoidal Rule', markersize=4)
        ax1.loglog(results['n_values'], results['errors_simp'], 'r-s', label='Simpson Rule', markersize=4)
        ax1.loglog(results['n_values'], results['errors_mc'], 'g-^', label='Monte Carlo', markersize=4)
        ax1.set_xlabel('Number of subdivisions (n)')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Error Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 오차 vs 계산 시간
        ax2.loglog(results['times_trap'], results['errors_trap'], 'b-o', label='Trapezoidal Rule', markersize=4)
        ax2.loglog(results['times_simp'], results['errors_simp'], 'r-s', label='Simpson Rule', markersize=4)
        ax2.loglog(results['times_mc'], results['errors_mc'], 'g-^', label='Monte Carlo', markersize=4)
        ax2.set_xlabel('Computation Time (seconds)')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Efficiency Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 함수 시각화
        x_plot = np.linspace(0, 1, 1000)
        y_plot = func_info['func'](x_plot)
        ax3.plot(x_plot, y_plot, 'k-', linewidth=1, label=f'{func_display_name}')
        ax3.set_xlabel('x')
        ax3.set_ylabel('f(x)')
        ax3.set_title('Function Visualization')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. 상대적 성능 비교
        mc_vs_trap = results['errors_mc'] / results['errors_trap']
        mc_vs_simp = results['errors_mc'] / results['errors_simp']
        
        ax4.semilogx(results['n_values'], mc_vs_trap, 'b-o', label='MC/Trapezoidal', markersize=3)
        ax4.semilogx(results['n_values'], mc_vs_simp, 'r-s', label='MC/Simpson', markersize=3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
        ax4.set_xlabel('Number of subdivisions (n)')
        ax4.set_ylabel('Relative Error Ratio')
        ax4.set_title('Monte Carlo Relative Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.1, 100)
        
        plt.tight_layout()
        plt.savefig(f'pseudo_random_analysis_{func_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self, all_results):
        """요약 보고서 생성"""
        report = "=" * 80 + "\n"
        report += "Pseudo Random 함수 분석 보고서\n"
        report += "=" * 80 + "\n\n"
        
        for func_name, results in all_results.items():
            if results is None:
                continue
                
            func_info = self.functions[func_name]
            exact_value = results['exact_value']
            
            report += f"함수: {func_info['name']}\n"
            report += f"설명: {func_info['description']}\n"
            report += f"정확한 값: {exact_value:.10f}\n"
            report += "-" * 50 + "\n"
            
            # 최종 오차
            final_n = results['n_values'][-1]
            final_error_trap = results['errors_trap'][-1]
            final_error_simp = results['errors_simp'][-1]
            final_error_mc = results['errors_mc'][-1]
            
            report += f"분할 수 n = {final_n}에서의 결과:\n"
            report += f"  사다리꼴 방법: 오차 = {final_error_trap:.3e}\n"
            report += f"  심슨 방법:     오차 = {final_error_simp:.3e}\n"
            report += f"  몬테카를로:    오차 = {final_error_mc:.3e}\n"
            
            # 상대적 성능 평가
            mc_vs_trap = final_error_mc / final_error_trap if final_error_trap > 0 else float('inf')
            mc_vs_simp = final_error_mc / final_error_simp if final_error_simp > 0 else float('inf')
            
            report += f"\n상대적 성능 (1.0 미만이면 몬테카를로가 유리):\n"
            report += f"  MC/Trapezoidal: {mc_vs_trap:.3f}\n"
            report += f"  MC/Simpson:     {mc_vs_simp:.3f}\n"
            
            # 몬테카를로 유리성 분석
            best_mc_vs_trap = np.min(results['errors_mc'] / results['errors_trap'])
            best_mc_vs_simp = np.min(results['errors_mc'] / results['errors_simp'])
            
            report += f"\n최고 성능 비교:\n"
            report += f"  최고 MC/Trapezoidal: {best_mc_vs_trap:.3f}\n"
            report += f"  최고 MC/Simpson:     {best_mc_vs_simp:.3f}\n"
            
            if best_mc_vs_trap < 0.5 or best_mc_vs_simp < 0.2:
                report += "\n>>> 몬테카를로가 일부 구간에서 경쟁력 있음! <<<\n"
            elif mc_vs_trap < 10 and mc_vs_simp < 50:
                report += "\n>>> 몬테카를로가 상대적으로 경쟁력 있음 <<<\n"
            else:
                report += "\n>>> 전통적 방법들이 우수함 <<<\n"
            
            report += "\n" + "=" * 80 + "\n\n"
        
        with open('pseudo_random_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report


def main():
    """메인 분석 함수"""
    analyzer = PseudoRandomAnalyzer()
    
    print("Pseudo Random 함수들의 수치적분 수렴성 분석")
    print("="*50)
    
    all_results = {}
    
    # 각 함수에 대해 분석 수행
    for func_name in analyzer.functions.keys():
        try:
            results = analyzer.analyze_convergence(func_name, max_points=3000)
            if results is not None:
                all_results[func_name] = results
                analyzer.plot_convergence_analysis(results, func_name)
        except Exception as e:
            print(f"Error analyzing {func_name}: {e}")
    
    # 요약 보고서 생성
    if all_results:
        print("\n요약 보고서 생성 중...")
        report = analyzer.generate_summary_report(all_results)
        print("\n" + report)
        
        print("\n분석 완료!")
        print("생성된 파일들:")
        print("- 각 함수별 그래프: pseudo_random_analysis_*.png")
        print("- 요약 보고서: pseudo_random_analysis_report.txt")
    else:
        print("\n분석할 수 있는 함수가 없습니다.")


if __name__ == "__main__":
    main() 
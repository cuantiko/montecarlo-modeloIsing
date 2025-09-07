import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def metropolis_ising(N, T, n_eq, n_steps, n_blocks=20):
    L = N
    # Inicialización de espines y energía inicial
    spins = np.random.choice([-1, 1], size=(L, L))
    E = 0
    for i in range(L):
        for j in range(L):
            E -= spins[i, j] * (spins[(i+1) % L, j] + spins[i, (j+1) % L])
    M = np.sum(spins)

    # Pre-calcula factores de Boltzmann
    dE_vals = [-8, -4, 0, 4, 8]
    boltz = {dE: np.exp(-dE / T) for dE in dE_vals}

    # Etapa de equilibrio
    for _ in range(n_eq):
        for _ in range(L*L):
            i = np.random.randint(L)
            j = np.random.randint(L)
            dE = 2 * spins[i, j] * (
                spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                spins[i, (j+1) % L] + spins[i, (j-1) % L]
            )
            if dE <= 0 or np.random.rand() < boltz.get(dE, np.exp(-dE/T)):
                spins[i, j] *= -1
                E += dE
                M += 2 * spins[i, j]

    # Bloques para estimar errores
    block_size = n_steps // n_blocks
    E_block = np.zeros(n_blocks)
    M_block = np.zeros(n_blocks)
    C_block = np.zeros(n_blocks)

    for b in range(n_blocks):
        E_acc = 0.0
        E2_acc = 0.0
        M_acc = 0.0
        for _ in range(block_size):
            # Paso de Monte Carlo
            for _ in range(L*L):
                i = np.random.randint(L)
                j = np.random.randint(L)
                dE = 2 * spins[i, j] * (
                    spins[(i+1) % L, j] + spins[(i-1) % L, j] +
                    spins[i, (j+1) % L] + spins[i, (j-1) % L]
                )
                if dE <= 0 or np.random.rand() < boltz.get(dE, np.exp(-dE/T)):
                    spins[i, j] *= -1
                    E += dE
                    M += 2 * spins[i, j]
            # Acumula observables
            E_acc += E
            E2_acc += E**2
            M_acc += abs(M)

        # Calcula medias por bloque
        avg_E = E_acc / block_size
        avg_E2 = E2_acc / block_size
        avg_M = M_acc / block_size

        # Normaliza por espín
        E_block[b] = avg_E / (N**2)
        M_block[b] = avg_M / (N**2)
        C_block[b] = (avg_E2 - avg_E**2) / (T**2 * N**2)

    # Media y error de cada magnitud
    E_mean = E_block.mean()
    M_mean = M_block.mean()
    C_mean = C_block.mean()
    err_E = E_block.std(ddof=1) / np.sqrt(n_blocks)
    err_M = M_block.std(ddof=1) / np.sqrt(n_blocks)
    err_C = C_block.std(ddof=1) / np.sqrt(n_blocks)

    return M_mean, E_mean, C_mean, err_M, err_E, err_C


if __name__ == '__main__':
    Ns = [8, 16, 32, 64]
    temps = np.linspace(1.5, 3.5, 10)
    n_eq = 10000
    n_steps = 10000

    # Almacenamiento de datos y errores
    M_data, E_data, C_data = [], [], []
    M_err, E_err, C_err = [], [], []

    for N in Ns:
        M_vals, E_vals, C_vals = [], [], []
        M_errs, E_errs, C_errs = [], [], []
        for T in temps:
            M, E, C, err_M, err_E, err_C = metropolis_ising(
                N, T, n_eq, n_steps)
            M_vals.append(M)
            E_vals.append(E)
            C_vals.append(C)
            M_errs.append(err_M)
            E_errs.append(err_E)
            C_errs.append(err_C)
            print(
                f"Done N={N}, T={T:.2f}: M={M:.4f}±{err_M:.4f}, E={E:.4f}±{err_E:.4f}, C={C:.4f}±{err_C:.4f}")
        M_data.append(M_vals)
        E_data.append(E_vals)
        C_data.append(C_vals)
        M_err.append(M_errs)
        E_err.append(E_errs)
        C_err.append(C_errs)

    colors = ['#FF00FF', '#00FFFF', '#FFA500', '#800080']
    Tc_theoretical = 2.269

    # GRAFICAS CON BARRAS DE ERROR
    plt.figure()
    for i, N in enumerate(Ns):
        plt.errorbar(temps, M_data[i], yerr=M_err[i],
                     color=colors[i], marker='o', capsize=3, label=f'N={N}')
    plt.axvline(Tc_theoretical, linestyle='--', color='gray')
    plt.xlabel('$T$')
    plt.ylabel('$m(T)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('magnetization.png')
    plt.close()

    plt.figure()
    for i, N in enumerate(Ns):
        plt.errorbar(temps, E_data[i], yerr=E_err[i],
                     color=colors[i], marker='s', capsize=3, label=f'N={N}')
    plt.axvline(Tc_theoretical, linestyle='--', color='gray')
    plt.xlabel('$T$')
    plt.ylabel('$e(T)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy.png')
    plt.close()

    plt.figure()
    for i, N in enumerate(Ns):
        plt.errorbar(temps, C_data[i], yerr=C_err[i],
                     color=colors[i], marker='^', capsize=3, label=f'N={N}')
    plt.axvline(Tc_theoretical, linestyle='--', color='gray')
    plt.xlabel('$T$')
    plt.ylabel('$c(T)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('specific_heat.png')
    plt.close()

    # Ajuste crítico
    C_largest = np.array(C_data[-1])
    Tc_est = temps[np.argmax(C_largest)]
    print(f"Estimated critical temperature Tc ≈ {Tc_est:.3f}")

    mask = temps < Tc_est
    x = Tc_est - temps[mask]
    y = np.array(M_data[-1])[mask]

    def mag_fit(x, A, beta):
        return A * x**beta

    popt, _ = curve_fit(mag_fit, x, y)
    A_fit, beta_est = popt
    print(f"Estimated beta exponent ≈ {beta_est:.3f}")

    with open('fit_results.txt', 'w') as f:
        f.write(f"Tc = {Tc_est:.6f}\n")
        f.write(f"beta = {beta_est:.6f}\n")

    print("All simulations and plots saved successfully.")

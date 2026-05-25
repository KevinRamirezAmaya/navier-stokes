import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ─────────────────────────────────────────────────────────────────
# Parámetros
# ─────────────────────────────────────────────────────────────────
Nx = 80       # índices i: 0..79
Ny = 8        # índices j: 0..7
N  = Nx * Ny  # 1280 total (640 por componente)

# Gradiente de presión virtual para sostener el flujo frente a la viscosidad = 1
dpdx = 0.04  

# ─────────────────────────────────────────────────────────────────
# Nueva Definición de Obstáculos (Esquinas especificadas)
# ─────────────────────────────────────────────────────────────────
def es_obstaculo(i, j):
    # Obstáculo 1: Esquinas (j=7, i=0) a (j=5, i=4) -> Superior izquierda
    if (0 <= i <= 4) and (5 <= j <= 7):
        return True
    # Obstáculo 2: Esquinas (j=0, i=36) a (j=2, i=43) -> Inferior central
    if (36 <= i <= 43) and (0 <= j <= 2):
        return True
    return False

def idx(i, j, comp=0):
    return comp * N + i * Ny + j

def unpack(x):
    vx = x[:N].reshape((Nx, Ny))
    vy = x[N:].reshape((Nx, Ny))
    return vx, vy

# ─────────────────────────────────────────────────────────────────
# Vector de residuos F(x)
# ─────────────────────────────────────────────────────────────────
def F_vector(x):
    vx, vy = unpack(x)
    F = np.zeros(2 * N)

    for i in range(Nx):
        for j in range(Ny):
            k_vx = idx(i, j, 0)
            k_vy = idx(i, j, 1)

            # 1. Zonas con obstáculos (Velocidad nula)
            if es_obstaculo(i, j):
                F[k_vx] = vx[i, j] - 0.0
                F[k_vy] = vy[i, j] - 0.0

            # 2. Entrada del canal (Condición de frontera izquierda)
            elif i == 0:
                F[k_vx] = vx[i, j] - 1.0  
                F[k_vy] = vy[i, j] - 0.0

            # 3. Paredes fijas (Superior e Inferior)
            elif j == 0 or j == Ny - 1:
                F[k_vx] = vx[i, j] - 0.0  
                F[k_vy] = vy[i, j] - 0.0

            # 4. Salida libre (Frontera derecha)
            elif i == Nx - 1:
                F[k_vx] = vx[i, j] - vx[i-1, j]
                F[k_vy] = vy[i, j] - vy[i-1, j]

            # 5. Volumen de control interno libre
            else:
                # Componente X (con dpdx acoplado)
                F[k_vx] = 0.25*(vx[i+1,j] + vx[i-1,j] + vx[i,j+1] + vx[i,j-1]
                       - 0.5*vx[i,j]*(vx[i+1,j] - vx[i-1,j])
                       - 0.5*vy[i,j]*(vx[i,j+1] - vx[i,j-1])) - vx[i,j] + dpdx

                # Componente Y
                F[k_vy] = 0.25*(vy[i+1,j] + vy[i-1,j] + vy[i,j+1] + vy[i,j-1]
                       - 0.5*vx[i,j]*(vy[i+1,j] - vy[i-1,j])
                       - 0.5*vy[i,j]*(vy[i,j+1] - vy[i,j-1])) - vy[i,j]
    return F

# ─────────────────────────────────────────────────────────────────
# Matriz Jacobiana J(x)
# ─────────────────────────────────────────────────────────────────
def jacobiana(x):
    vx, vy = unpack(x)
    J = lil_matrix((2*N, 2*N))

    for i in range(Nx):
        for j in range(Ny):
            k_vx = idx(i, j, 0)
            k_vy = idx(i, j, 1)

            if es_obstaculo(i, j) or (i == 0) or (j == 0 or j == Ny - 1):
                J[k_vx, idx(i, j, 0)] = 1.0
                J[k_vy, idx(i, j, 1)] = 1.0

            elif i == Nx - 1:
                J[k_vx, idx(i, j, 0)] = 1.0
                J[k_vx, idx(i-1, j, 0)] = -1.0
                J[k_vy, idx(i, j, 1)] = 1.0
                J[k_vy, idx(i-1, j, 1)] = -1.0

            else:
                # Fila de vx
                J[k_vx, idx(i,   j,   0)] = 0.25*(-0.5*(vx[i+1,j]-vx[i-1,j])) - 1
                J[k_vx, idx(i+1, j,   0)] = 0.25*(1 - 0.5*vx[i,j])
                J[k_vx, idx(i-1, j,   0)] = 0.25*(1 + 0.5*vx[i,j])
                J[k_vx, idx(i,   j+1, 0)] = 0.25*(1 - 0.5*vy[i,j])
                J[k_vx, idx(i,   j-1, 0)] = 0.25*(1 + 0.5*vy[i,j])
                J[k_vx, idx(i,   j,   1)] = 0.25*(-0.5*(vx[i,j+1]-vx[i,j-1]))

                # Fila de vy
                J[k_vy, idx(i,   j,   1)] = 0.25*(-0.5*(vy[i,j+1]-vy[i,j-1])) - 1
                J[k_vy, idx(i+1, j,   1)] = 0.25*(1 - 0.5*vx[i,j])
                J[k_vy, idx(i-1, j,   1)] = 0.25*(1 + 0.5*vx[i,j])
                J[k_vy, idx(i,   j+1, 1)] = 0.25*(1 - 0.5*vy[i,j])
                J[k_vy, idx(i,   j-1, 1)] = 0.25*(1 + 0.5*vy[i,j])
                J[k_vy, idx(i,   j,   0)] = 0.25*(-0.5*(vy[i+1,j]-vy[i-1,j]))

    return J.tocsr()

# ─────────────────────────────────────────────────────────────────
# Solucionador Newton-Raphson
# ─────────────────────────────────────────────────────────────────
def newton_raphson(tol=1e-6, max_iter=50):
    x = np.zeros(2 * N)  
    historial = []

    print(f"{'Iter':>5}  {'||F||':>12}")
    print("-" * 20)

    for n in range(max_iter):
        Fval = F_vector(x)
        norma = np.linalg.norm(Fval)
        historial.append(norma)
        print(f"{n:>5}  {norma:>12.6e}")

        if norma < tol:
            print(f"\nConvergió con éxito en {n} iteraciones.")
            break

        J = jacobiana(x)
        dx = spsolve(J, -Fval)
        x = x + dx
    else:
        print("\nNo se alcanzó la convergencia estricta.")

    return x, historial

# ─────────────────────────────────────────────────────────────────
# Renderizado de Gráficas y Parches
# ─────────────────────────────────────────────────────────────────
def graficar(x, historial):
    vx, vy = unpack(x)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Canal Vx
    im0 = axes[0].imshow(vx.T, origin='lower', aspect='auto', cmap='hot')
    axes[0].set_title(r'Campo de velocidad $v_x$')
    axes[0].set_xlabel('i (dirección x)')
    axes[0].set_ylabel('j (dirección y)')
    plt.colorbar(im0, ax=axes[0], label=r'$v_x$')

    # Canal Vy
    im1 = axes[1].imshow(vy.T, origin='lower', aspect='auto', cmap='coolwarm')
    axes[1].set_title(r'Campo de velocidad $v_y$')
    axes[1].set_xlabel('i (dirección x)')
    axes[1].set_ylabel('j (dirección y)')
    plt.colorbar(im1, ax=axes[1], label=r'$v_y$')

    # Obstáculo 1: i de 0 a 4 (ancho=5), j de 5 a 7 (alto=3). Base inferior izquierda en (-0.5, 4.5)
    obs1_vx = patches.Rectangle((-0.5, 4.5), 5, 3, color='gray', hatch='//', alpha=0.8, edgecolor='black')
    obs1_vy = patches.Rectangle((-0.5, 4.5), 5, 3, color='gray', hatch='//', alpha=0.8, edgecolor='black')
    
    # Obstáculo 2: i de 36 a 43 (ancho=8), j de 0 a 2 (alto=3). Base inferior izquierda en (35.5, -0.5)
    obs2_vx = patches.Rectangle((35.5, -0.5), 8, 3, color='gray', hatch='//', alpha=0.8, edgecolor='black')
    obs2_vy = patches.Rectangle((35.5, -0.5), 8, 3, color='gray', hatch='//', alpha=0.8, edgecolor='black')

    # Inyección de los bloques en los ejes correspondientes
    axes[0].add_patch(obs1_vx)
    axes[0].add_patch(obs2_vx)
    axes[1].add_patch(obs1_vy)
    axes[1].add_patch(obs2_vy)

    # Gráfica del Historial de Residuos
    axes[2].semilogy(historial, 'b-o', markersize=4)
    axes[2].set_title('Convergencia Newton-Raphson')
    axes[2].set_xlabel('Iteración')
    axes[2].set_ylabel(r'$\|\mathbf{F}\|_2$')
    axes[2].grid(True, which='both', alpha=0.4)

    plt.tight_layout()
    plt.savefig('resultados_navier_stokes_nuevas_esquinas.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    x_sol, hist = newton_raphson(tol=1e-6, max_iter=50)
    graficar(x_sol, hist)
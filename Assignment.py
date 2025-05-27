import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import simpson

# defined class MobiusStrip
class MobiusStrip:
    def __init__(self, R, w, n):
        self.R = R
        self.w = w
        self.n = n
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._compute_mesh()

    def _compute_mesh(self):
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def surface_area(self):
       
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]
        dXdu, dXdv = np.gradient(self.X, du, dv)
        dYdu, dYdv = np.gradient(self.Y, du, dv)
        dZdu, dZdv = np.gradient(self.Z, du, dv)

       
        Nx = dYdu * dZdv - dZdu * dYdv
        Ny = dZdu * dXdv - dXdu * dZdv
        Nz = dXdu * dYdv - dYdu * dXdv
        dA = np.sqrt(Nx**2 + Ny**2 + Nz**2)

        # Integrate over the surface
        area = simpson(simpson(dA, self.v), self.u)
        return area

    def edge_length(self):
      
        lengths = []
        for sign in [-1, 1]:
            v = sign * self.w / 2
            x = (self.R + v * np.cos(self.u / 2)) * np.cos(self.u)
            y = (self.R + v * np.cos(self.u / 2)) * np.sin(self.u)
            z = v * np.sin(self.u / 2)
            dx = np.gradient(x, self.u)
            dy = np.gradient(y, self.u)
            dz = np.gradient(z, self.u)
            ds = np.sqrt(dx**2 + dy**2 + dz**2)
            lengths.append(simpson(ds, self.u))
        return sum(lengths)

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, color='lightblue', edgecolor='k', alpha=0.8)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    strip = MobiusStrip(R=1, w=0.4, n=200)
    print("Surface Area:", strip.surface_area())
    print("Edge Length:", strip.edge_length())
    strip.plot()

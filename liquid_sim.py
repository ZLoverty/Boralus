import numpy as np
import pyvtk
import os

# Constants
WIDTH, HEIGHT, DEPTH = 100, 100, 100
DT = 1.0e-6 # seconds
DX = 1.0e-4 # meter
T = 0.001
GRAVITY = 9.8 * DT**2 / DX  # Gravity acceleration
CUTOFF = 1.0
TENSION_CUTOFF = 3.0
A = 10*GRAVITY  # Conservative force coefficient
GAMMA = 10*GRAVITY  # Dissipative force coefficient
SIGMA_DPD = np.sqrt(2 * GAMMA)  # Random force coefficient
DAMPING_COEFFICIENT = 0.5  # Damping coefficient for the bottom boundary
ATTRACTIVE_FORCE = 1.0  # Short-range attractive force near the bottom boundary
PARALLEL_RESISTANCE = 0.5  # Resistance to motion parallel to the bottom boundary

# Initial positions
def generate_lattice(x_min, x_max, y_min, y_max, z_min, z_max, spacing):
    x_values = np.arange(x_min, x_max, spacing)
    y_values = np.arange(y_min, y_max, spacing)
    z_values = np.arange(z_min, z_max, spacing)
    x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values)
    xyz = np.c_[x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]
    return xyz

XYZ = generate_lattice(48, 53, 0, 5, 48, 53, CUTOFF)

class ParticleSimulation:
    def __init__(self, xyz, width, height, depth):
        
        self.width = width
        self.height = height
        self.depth = depth
        self.num_particles = xyz.shape[0]
        self.positions = xyz
        self.velocities = np.random.uniform(-.1*DT/DX, .1*DT/DX, (self.num_particles, 3))
        self.forces = np.zeros((self.num_particles, 3))

    def calculate_forces(self):
        self.forces.fill(0)
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx, dy, dz = self.positions[i] - self.positions[j]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                force_c = np.zeros(3)
                force_d = np.zeros(3)
                force_r = np.zeros(3)
                # Conservative force
                if distance < TENSION_CUTOFF:
                    force_c = A * (1 - distance) * np.array([dx, dy, dz]) / distance
                

                if distance < CUTOFF:
                    # Relative velocity
                    dvx, dvy, dvz = self.velocities[i] - self.velocities[j]

                    # Dissipative force
                    force_d = -GAMMA * (1-distance)**2 * (dvx * dx + dvy * dy + dvz * dz) / distance**2 * np.array([dx, dy, dz])

                    # Random force
                    theta = np.random.normal(0, 1)
                    force_r = SIGMA_DPD * max(1 - distance, 0) * theta * np.array([dx, dy, dz]) / distance

                # Total force
                force = force_c + force_d + force_r

                self.forces[i] += force
                self.forces[j] -= force

        # Apply gravity
        self.forces[:, 1] -= GRAVITY

    def update(self):
        self.calculate_forces()
        self.velocities += self.forces 
        self.positions += self.velocities 

        # Boundary conditions
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, self.width)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, self.height)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0, self.depth)

        for i in range(self.num_particles):
            if self.positions[i, 0] == 0 or self.positions[i, 0] == self.width:
                self.velocities[i, 0] *= -DAMPING_COEFFICIENT
            if self.positions[i, 1] == 0 or self.positions[i, 1] == self.height:
                self.velocities[i, 1] *= -DAMPING_COEFFICIENT
            if self.positions[i, 2] == 0 or self.positions[i, 2] == self.depth:
                self.velocities[i, 2] *= -DAMPING_COEFFICIENT

            # Damping boundary at the bottom
            if self.positions[i, 1] <= 1:
                # Short-range attraction
                self.forces[i, 1] += ATTRACTIVE_FORCE * (1 - self.positions[i, 1])

                # Reduce speed (damping)
                self.velocities[i] *= DAMPING_COEFFICIENT

                # Resist motion parallel to the bottom boundary
                self.velocities[i, 0] *= PARALLEL_RESISTANCE
                self.velocities[i, 2] *= PARALLEL_RESISTANCE

    def save_vtk(self, step, save_folder):
        points = self.positions.tolist()
        vtk_data = pyvtk.VtkData(
            pyvtk.UnstructuredGrid(points),
            "Particle simulation step %d" % step
        )
        vtk_data.tofile(os.path.join(save_folder, f"data_{step:04d}.vtk"))

    def run(self, steps, save_folder, print_step=1, save_step=10):
        self.create_log(save_folder)
        for step in range(steps):
            print(f"Step {step + 1}")
            self.update()
            if step % print_step == 0:
                ke = self.compute_energy()
                with open(os.path.join(save_folder, "log.txt"), "a") as f:
                    f.write(f"{step:<9.4f}{ke:<9.1e}\n")
        
            if step % save_step == 0:
                self.save_vtk(step, save_folder)

    def create_log(self, save_folder):
        """Create a log txt file to store energy"""
        with open(os.path.join(save_folder, "log.txt"), "w") as f:
            f.write(f"{'Time':10s}{'Energy':10s}\n")

    def compute_energy(self):
        ke = np.sum(self.velocities**2) / 3
        return ke
if __name__ == "__main__":
    simulation = ParticleSimulation(XYZ, WIDTH, HEIGHT, DEPTH)
    simulation.run(np.floor(T/DT).astype(int), r"C:\Users\zl948\Documents\tmp")
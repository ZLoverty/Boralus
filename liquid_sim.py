import numpy as np
import pyvtk
import os

# Constants
WIDTH, HEIGHT, DEPTH = 100, 100, 100
NUM_PARTICLES = 100
RADIUS = 5
GRAVITY = -100
FRICTION = 0.99
SIGMA = 2 * RADIUS  # Distance at which the potential is zero
EPSILON = 1.0  # Depth of the potential well
CUTOFF = 2**(1/6) * SIGMA  # Cutoff distance for WCA potential
DT = 0.01  # Time step
A = 25.0  # Conservative force coefficient
GAMMA = 10.5  # Dissipative force coefficient
SIGMA_DPD = np.sqrt(2 * GAMMA * 1.0 / DT)  # Random force coefficient
DAMPING_COEFFICIENT = 0.5  # Damping coefficient for the bottom boundary
ATTRACTIVE_FORCE = 10.0  # Short-range attractive force near the bottom boundary
PARALLEL_RESISTANCE = 0.5  # Resistance to motion parallel to the bottom boundary

class ParticleSimulation:
    def __init__(self, num_particles, width, height, depth):
        self.num_particles = num_particles
        self.width = width
        self.height = height
        self.depth = depth
        self.positions = np.random.uniform(RADIUS, [width - RADIUS, height - RADIUS, depth - RADIUS], (num_particles, 3))
        self.velocities = np.random.uniform(-1, 1, (num_particles, 3))
        self.forces = np.zeros((num_particles, 3))

    def calculate_forces(self):
        self.forces.fill(0)
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                dx, dy, dz = self.positions[j] - self.positions[i]
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                if distance < CUTOFF:
                    # Conservative force
                    force_c = A * (1 - distance / CUTOFF) * np.array([dx, dy, dz]) / distance

                    # Relative velocity
                    dvx, dvy, dvz = self.velocities[j] - self.velocities[i]

                    # Dissipative force
                    force_d = -GAMMA * (dvx * dx + dvy * dy + dvz * dz) / distance**2 * np.array([dx, dy, dz])

                    # Random force
                    theta = np.random.normal(0, 1)
                    force_r = SIGMA_DPD * theta / np.sqrt(DT) * np.array([dx, dy, dz]) / distance

                    # Total force
                    force = force_c + force_d + force_r

                    self.forces[i] += force
                    self.forces[j] -= force

        # Apply gravity
        self.forces[:, 2] += GRAVITY

    def update(self):
        self.calculate_forces()
        self.velocities += self.forces * DT
        self.positions += self.velocities * DT

        # Boundary conditions
        self.positions[:, 0] = np.clip(self.positions[:, 0], RADIUS, self.width - RADIUS)
        self.positions[:, 1] = np.clip(self.positions[:, 1], RADIUS, self.height - RADIUS)
        self.positions[:, 2] = np.clip(self.positions[:, 2], RADIUS, self.depth - RADIUS)

        for i in range(self.num_particles):
            if self.positions[i, 0] == RADIUS or self.positions[i, 0] == self.width - RADIUS:
                self.velocities[i, 0] *= -1
            if self.positions[i, 1] == RADIUS or self.positions[i, 1] == self.height - RADIUS:
                self.velocities[i, 1] *= -1
            if self.positions[i, 2] == RADIUS or self.positions[i, 2] == self.depth - RADIUS:
                self.velocities[i, 2] *= -1

        # Damping boundary at the bottom
        if self.positions[i, 1] <= RADIUS:
            # Short-range attraction
            self.forces[i, 1] += ATTRACTIVE_FORCE * (RADIUS - self.positions[i, 1]) / RADIUS

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

    def run(self, steps, save_folder):
        for step in range(steps):
            self.update()
            self.save_vtk(step, save_folder)
            print(f"Step {step + 1}")

if __name__ == "__main__":
    simulation = ParticleSimulation(NUM_PARTICLES, WIDTH, HEIGHT, DEPTH)
    simulation.run(100, r"C:\Users\zl948\Documents\tmp")
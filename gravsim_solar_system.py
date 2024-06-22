import pygame
import math
import collections
import random
import numpy as np

# Initialize Pygame
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LLM Solar System Simulation")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
RED = (255, 0, 0)
LIGHT_BLUE = (173, 216, 230)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 139)
GRAY = (169, 169, 169)
BROWN = (165, 42, 42)

# Constants
G = 6.67430e-11
SCALE = 2e-10
BASE_TIME_STEP = 3600  # 1 hour
SUN_MASS = 1.989e30
SUN_RADIUS = 696340000

# Planet data
PLANETS = [
    {"name": "Mercury", "mass": 3.285e23, "distance": 57.9e9, "radius": 2439700, "color": GRAY},
    {"name": "Venus", "mass": 4.867e24, "distance": 108.2e9, "radius": 6051800, "color": ORANGE},
    {"name": "Earth", "mass": 5.972e24, "distance": 149.6e9, "radius": 6371000, "color": BLUE},
    {"name": "Mars", "mass": 6.39e23, "distance": 227.9e9, "radius": 3389500, "color": RED},
    {"name": "Jupiter", "mass": 1.898e27, "distance": 778.5e9, "radius": 69911000, "color": LIGHT_BLUE},
    {"name": "Saturn", "mass": 5.683e26, "distance": 1.434e12, "radius": 58232000, "color": YELLOW},
    {"name": "Uranus", "mass": 8.681e25, "distance": 2.871e12, "radius": 25362000, "color": LIGHT_BLUE},
    {"name": "Neptune", "mass": 1.024e26, "distance": 4.495e12, "radius": 24622000, "color": DARK_BLUE}
]

# Moon data
MOONS = {
    "Earth": [{"name": "Moon", "mass": 7.34767309e22, "distance": 384400000, "radius": 1737100, "color": GRAY}],
    "Mars": [
        {"name": "Phobos", "mass": 1.0659e16, "distance": 9376000, "radius": 11266, "color": GRAY},
        {"name": "Deimos", "mass": 1.4762e15, "distance": 23463200, "radius": 6200, "color": GRAY}
    ],
    "Jupiter": [
        {"name": "Io", "mass": 8.94e22, "distance": 421700000, "radius": 1821600, "color": YELLOW},
        {"name": "Europa", "mass": 4.80e22, "distance": 671100000, "radius": 1560800, "color": LIGHT_BLUE},
        {"name": "Ganymede", "mass": 1.48e23, "distance": 1070400000, "radius": 2634100, "color": GRAY},
        {"name": "Callisto", "mass": 1.08e23, "distance": 1882700000, "radius": 2410300, "color": BROWN}
    ],
    "Saturn": [
        {"name": "Titan", "mass": 1.34552e23, "distance": 1221870000, "radius": 2574730, "color": ORANGE},
        {"name": "Rhea", "mass": 2.306e21, "distance": 527108000, "radius": 763800, "color": GRAY}
    ],
    "Uranus": [
        {"name": "Titania", "mass": 3.40e21, "distance": 435910000, "radius": 788400, "color": GRAY},
        {"name": "Oberon", "mass": 3.08e21, "distance": 583520000, "radius": 761400, "color": GRAY}
    ],
    "Neptune": [
        {"name": "Triton", "mass": 2.14e22, "distance": 354759000, "radius": 1353400, "color": GRAY}
    ]
}

# Simulation parameters
ENERGY_DEVIATION_THRESHOLD = 0.001
MAX_SUBSTEPS = 10
TRAIL_OPTIONS = ["Full", "Partial", "None"]

# Generate background stars
stars = [(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(200)]

class CelestialBody:
    def __init__(self, name, mass, x, y, vx, vy, radius, color, is_moon=False):
        self.name = name
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.color = color
        self.is_moon = is_moon
        self.trace = collections.deque(maxlen=1000)
        self.trail_option = "Full"

    def update_color(self):
        if self.name.startswith("Sun") or self.name.startswith("Star_") or self.name.startswith("Merged_"):
            temperature = 5778 * (self.mass / SUN_MASS) ** 0.505
            r = min(255, max(0, int(255 * (temperature - 1000) / 29000)))
            g = min(255, max(0, int(255 * (temperature - 1000) / 6000)))
            b = min(255, max(0, int(255 * (temperature - 1000) / 3000)))
            self.color = (r, g, b)

    def draw(self, screen, center_x, center_y, zoom, font):
        self.update_color()
        x = int(center_x + self.x * SCALE * zoom)
        y = int(center_y + self.y * SCALE * zoom)
        radius = max(2, int(self.radius * SCALE * zoom))
        pygame.draw.circle(screen, self.color, (x, y), radius)

        if zoom > 0.5 or self.name.startswith("Sun") or self.name.startswith("Star_") or self.name.startswith("Merged_"):
            text = font.render(self.name, True, WHITE)
            screen.blit(text, (x + radius + 5, y - 10))

        if self.trail_option != "None" and len(self.trace) > 1:
            if self.trail_option == "Partial":
                trace_length = min(100, len(self.trace))
                scaled_trace = [(int(center_x + px * SCALE * zoom), int(center_y + py * SCALE * zoom)) 
                                for px, py in list(self.trace)[-trace_length:]]
            else:  # Full
                scaled_trace = [(int(center_x + px * SCALE * zoom), int(center_y + py * SCALE * zoom)) 
                                for px, py in self.trace]
            pygame.draw.lines(screen, self.color, False, scaled_trace, 1)

    def update_trail(self):
        if self.trail_option != "None":
            self.trace.append((self.x, self.y))

def calculate_initial_velocity(body, all_bodies):
    total_fx = 0
    total_fy = 0
    for other_body in all_bodies:
        if body != other_body:
            dx = other_body.x - body.x
            dy = other_body.y - body.y
            r = math.sqrt(dx**2 + dy**2)
            f = G * body.mass * other_body.mass / (r**2 + 1e10)
            total_fx += f * dx / r
            total_fy += f * dy / r
    
    r = math.sqrt(body.x**2 + body.y**2)
    v_circular = math.sqrt(G * SUN_MASS / r)
    vx = -v_circular * body.y / r
    vy = v_circular * body.x / r
    
    return vx, vy

def calculate_total_energy(bodies):
    total_energy = 0
    for i, body in enumerate(bodies):
        kinetic_energy = 0.5 * body.mass * (body.vx**2 + body.vy**2)
        potential_energy = 0
        for j, other in enumerate(bodies):
            if i != j:
                r = math.sqrt((body.x - other.x)**2 + (body.y - other.y)**2)
                potential_energy -= G * body.mass * other.mass / r
        total_energy += kinetic_energy + potential_energy
    return total_energy

def rk4_step(bodies, dt, initial_energy):
    def acceleration(bodies):
        acc = np.zeros((len(bodies), 2))
        for i, body in enumerate(bodies):
            for j, other in enumerate(bodies):
                if i != j:
                    r = np.array([other.x - body.x, other.y - body.y])
                    r_mag = np.linalg.norm(r)
                    acc[i] += G * other.mass * r / (r_mag**3 + 1e-10)  # Softening
        return acc

    pos = np.array([[body.x, body.y] for body in bodies])
    vel = np.array([[body.vx, body.vy] for body in bodies])

    k1v = acceleration(bodies) * dt
    k1r = vel * dt

    temp_bodies = [CelestialBody(b.name, b.mass, b.x + k1r[i][0]/2, b.y + k1r[i][1]/2, b.vx + k1v[i][0]/2, b.vy + k1v[i][1]/2, b.radius, b.color, b.is_moon) for i, b in enumerate(bodies)]
    k2v = acceleration(temp_bodies) * dt
    k2r = (vel + k1v/2) * dt

    temp_bodies = [CelestialBody(b.name, b.mass, b.x + k2r[i][0]/2, b.y + k2r[i][1]/2, b.vx + k2v[i][0]/2, b.vy + k2v[i][1]/2, b.radius, b.color, b.is_moon) for i, b in enumerate(bodies)]
    k3v = acceleration(temp_bodies) * dt
    k3r = (vel + k2v/2) * dt

    temp_bodies = [CelestialBody(b.name, b.mass, b.x + k3r[i][0], b.y + k3r[i][1], b.vx + k3v[i][0], b.vy + k3v[i][1], b.radius, b.color, b.is_moon) for i, b in enumerate(bodies)]
    k4v = acceleration(temp_bodies) * dt
    k4r = (vel + k3v) * dt

    pos_new = pos + (k1r + 2*k2r + 2*k3r + k4r) / 6
    vel_new = vel + (k1v + 2*k2v + 2*k3v + k4v) / 6

    # Safety check: limit maximum position change
    max_pos_change = 1e10  # 10 billion meters (about 1/15 AU)
    pos_change = pos_new - pos
    pos_change_magnitude = np.linalg.norm(pos_change, axis=1)
    scale_factors = np.minimum(1, max_pos_change / pos_change_magnitude)
    pos_new = pos + pos_change * scale_factors[:, np.newaxis]

    for i, body in enumerate(bodies):
        body.x, body.y = pos_new[i]
        body.vx, body.vy = vel_new[i]
        body.update_trail()

    # Energy conservation check
    current_energy = calculate_total_energy(bodies)
    energy_ratio = current_energy / initial_energy
    
    if abs(energy_ratio - 1) > ENERGY_DEVIATION_THRESHOLD:
        try:
            correction_factor = math.sqrt(1 / energy_ratio)
            for body in bodies:
                body.vx *= correction_factor
                body.vy *= correction_factor
        except ValueError:
            print(f"Warning: Energy conservation failed. Ratio: {energy_ratio}")

    return current_energy, bodies

def initialize_solar_system():
    sun = CelestialBody("Sun", SUN_MASS, 0, 0, 0, 0, SUN_RADIUS, YELLOW)
    all_bodies = [sun]

    for planet_data in PLANETS:
        distance = planet_data["distance"]
        angle = random.uniform(0, 2 * math.pi)
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        planet = CelestialBody(planet_data["name"], planet_data["mass"], x, y, 0, 0, planet_data["radius"], planet_data["color"])
        all_bodies.append(planet)

    for body in all_bodies[1:]:  # Skip Sun
        vx, vy = calculate_initial_velocity(body, all_bodies)
        body.vx, body.vy = vx, vy

    for planet in all_bodies[1:]:  # Skip Sun
        if planet.name in MOONS:
            for moon_data in MOONS[planet.name]:
                angle = random.uniform(0, 2 * math.pi)
                distance = moon_data["distance"]
                x = planet.x + math.cos(angle) * distance
                y = planet.y + math.sin(angle) * distance
                
                moon = CelestialBody(moon_data["name"], moon_data["mass"], x, y, planet.vx, planet.vy, moon_data["radius"], moon_data["color"], is_moon=True)
                
                v_orbit = math.sqrt(G * planet.mass / distance)
                moon.vx += -math.sin(angle) * v_orbit
                moon.vy += math.cos(angle) * v_orbit
                
                all_bodies.append(moon)

    for body in all_bodies:
        body.trail_option = "Full"

    return all_bodies

def create_new_star(x, y, all_bodies, zoom, center_x, center_y):
    new_star_mass = SUN_MASS * random.uniform(0.1, 2.0)
    new_star_radius = SUN_RADIUS * (new_star_mass / SUN_MASS) ** 0.5
    new_star_color = (random.randint(200, 255), random.randint(150, 255), random.randint(0, 100))
    
    sim_x = (x - center_x) / (SCALE * zoom)
    sim_y = (y - center_y) / (SCALE * zoom)
    
    new_star = CelestialBody(f"Star_{len(all_bodies)}", new_star_mass, sim_x, sim_y, 0, 0, new_star_radius, new_star_color)
    new_star.trail_option = all_bodies[0].trail_option
    all_bodies.append(new_star)
    
    return all_bodies

def check_collisions(bodies):
    merged_bodies = []
    removed_indices = set()
    for i, body1 in enumerate(bodies):
        if i in removed_indices:
            continue
        for j, body2 in enumerate(bodies[i+1:], start=i+1):
            if j in removed_indices:
                continue
            dx = body1.x - body2.x
            dy = body1.y - body2.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < (body1.radius + body2.radius):
                new_mass = body1.mass + body2.mass
                new_x = (body1.x * body1.mass + body2.x * body2.mass) / new_mass
                new_y = (body1.y * body1.mass + body2.y * body2.mass) / new_mass
                new_vx = (body1.vx * body1.mass + body2.vx * body2.mass) / new_mass
                new_vy = (body1.vy * body1.mass + body2.vy * body2.mass) / new_mass
                new_radius = ((body1.radius**3 + body2.radius**3)**(1/3))
                new_color = tuple(int((c1 * body1.mass + c2 * body2.mass) / new_mass) for c1, c2 in zip(body1.color, body2.color))
                new_name = f"Merged_{body1.name}_{body2.name}"
                
                new_body = CelestialBody(new_name, new_mass, new_x, new_y, new_vx, new_vy, new_radius, new_color)
                new_body.trail_option = body1.trail_option
                merged_bodies.append(new_body)
                removed_indices.add(i)
                removed_indices.add(j)
                break
    
    updated_bodies = [body for i, body in enumerate(bodies) if i not in removed_indices] + merged_bodies
    return updated_bodies

def draw_ui(screen, font, selected_body, editing_body, current_trail_option, simulation_time, time_multiplier, all_bodies, dropdown_open, center_x, center_y, zoom, spawn_star_mode):
    # Draw buttons
    restart_button = pygame.Rect(50, HEIGHT - 180, 100, 30)
    pygame.draw.rect(screen, GRAY, restart_button)
    restart_text = font.render("Restart", True, BLACK)
    screen.blit(restart_text, (restart_button.x + 10, restart_button.y + 5))

    reset_zoom_button = pygame.Rect(160, HEIGHT - 180, 100, 30)
    pygame.draw.rect(screen, GRAY, reset_zoom_button)
    reset_zoom_text = font.render("Reset Zoom", True, BLACK)
    screen.blit(reset_zoom_text, (reset_zoom_button.x + 5, reset_zoom_button.y + 5))

    trail_button = pygame.Rect(270, HEIGHT - 180, 100, 30)
    pygame.draw.rect(screen, GRAY, trail_button)
    trail_text = font.render(f"Trail: {TRAIL_OPTIONS[current_trail_option]}", True, BLACK)
    screen.blit(trail_text, (trail_button.x + 5, trail_button.y + 5))

    spawn_star_button = pygame.Rect(380, HEIGHT - 180, 100, 30)
    spawn_star_color = LIGHT_BLUE if spawn_star_mode else GRAY
    pygame.draw.rect(screen, spawn_star_color, spawn_star_button)
    spawn_star_text = font.render("Spawn Star", True, BLACK)
    screen.blit(spawn_star_text, (spawn_star_button.x + 5, spawn_star_button.y + 5))

    # Draw dropdown for body selection
    dropdown_rect = pygame.Rect(WIDTH - 250, HEIGHT - 50, 200, 30)
    pygame.draw.rect(screen, GRAY, dropdown_rect)
    dropdown_text = font.render(f"Center on: {selected_body}", True, BLACK)
    screen.blit(dropdown_text, (dropdown_rect.x + 10, dropdown_rect.y + 5))

    if dropdown_open:
        non_moon_bodies = [b for b in all_bodies if not b.is_moon]
        for i, body in enumerate(reversed(non_moon_bodies)):
            option_rect = pygame.Rect(dropdown_rect.x, dropdown_rect.y - (i+1)*30, 200, 30)
            pygame.draw.rect(screen, GRAY, option_rect)
            option_text = font.render(body.name, True, BLACK)
            screen.blit(option_text, (option_rect.x + 10, option_rect.y + 5))

    # Draw speed slider
    slider_rect = pygame.Rect(50, HEIGHT - 50, 200, 20)
    pygame.draw.rect(screen, GRAY, slider_rect)
    slider_pos = int(slider_rect.x + (time_multiplier - 1) * 10)
    pygame.draw.rect(screen, WHITE, (slider_pos, slider_rect.y, 10, 20))
    speed_text = font.render(f"Speed: {time_multiplier:.1f}x", True, WHITE)
    screen.blit(speed_text, (slider_rect.x, slider_rect.y - 25))

    # Display simulation time
    days = simulation_time / (24 * 3600)
    years = days / 365.25
    time_text = font.render(f"Simulation Time: {years:.2f} years", True, WHITE)
    screen.blit(time_text, (50, HEIGHT - 110))

    # Draw edit panel if a body is being edited
    if editing_body:
        body_x = int(center_x + editing_body.x * SCALE * zoom)
        body_y = int(center_y + editing_body.y * SCALE * zoom)
        
        panel_x = min(body_x + 50, WIDTH - 260)
        panel_y = max(body_y - 75, 10)
        
        edit_panel = pygame.Rect(panel_x, panel_y, 250, 180)
        pygame.draw.rect(screen, GRAY, edit_panel)
        
        title = font.render(f"Editing {editing_body.name}", True, BLACK)
        screen.blit(title, (edit_panel.x + 10, edit_panel.y + 10))
        
        mass_text = font.render(f"Mass: {editing_body.mass:.2e} kg", True, BLACK)
        screen.blit(mass_text, (edit_panel.x + 10, edit_panel.y + 40))
        
        radius_text = font.render(f"Radius: {editing_body.radius:.2e} m", True, BLACK)
        screen.blit(radius_text, (edit_panel.x + 10, edit_panel.y + 70))
        
        instructions1 = font.render("Up/Down: Adjust mass", True, BLACK)
        screen.blit(instructions1, (edit_panel.x + 10, edit_panel.y + 100))
        
        instructions2 = font.render("Left/Right: Adjust radius", True, BLACK)
        screen.blit(instructions2, (edit_panel.x + 10, edit_panel.y + 130))
        
        instructions3 = font.render("ESC: Exit editing mode", True, BLACK)
        screen.blit(instructions3, (edit_panel.x + 10, edit_panel.y + 160))

    return restart_button, reset_zoom_button, trail_button, spawn_star_button, dropdown_rect, slider_rect

def main():
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    all_bodies = initialize_solar_system()
    initial_energy = calculate_total_energy(all_bodies)

    zoom = 1
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    selected_body = "Sun"
    editing_body = None
    spawn_star_mode = False
    simulation_time = 0
    current_trail_option = 0
    time_multiplier = 1
    dropdown_open = False
    slider_dragging = False
    dragging = False
    drag_start = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                zoom *= 1.1 if event.y > 0 else 0.9
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()
                    if restart_button.collidepoint(mouse_pos):
                        all_bodies = initialize_solar_system()
                        simulation_time = 0
                        selected_body = "Sun"
                        initial_energy = calculate_total_energy(all_bodies)
                    elif reset_zoom_button.collidepoint(mouse_pos):
                        zoom = 1
                        center_x, center_y = WIDTH // 2, HEIGHT // 2
                    elif trail_button.collidepoint(mouse_pos):
                        current_trail_option = (current_trail_option + 1) % len(TRAIL_OPTIONS)
                        for body in all_bodies:
                            body.trail_option = TRAIL_OPTIONS[current_trail_option]
                            if body.trail_option == "None":
                                body.trace.clear()
                    elif spawn_star_button.collidepoint(mouse_pos):
                        spawn_star_mode = not spawn_star_mode
                    elif dropdown_rect.collidepoint(mouse_pos):
                        dropdown_open = not dropdown_open
                    elif dropdown_open:
                        non_moon_bodies = [b for b in all_bodies if not b.is_moon]
                        for i, body in enumerate(reversed(non_moon_bodies)):
                            option_rect = pygame.Rect(dropdown_rect.x, dropdown_rect.y - (i+1)*30, 200, 30)
                            if option_rect.collidepoint(mouse_pos):
                                selected_body = body.name
                                dropdown_open = False
                                break
                    elif slider_rect.collidepoint(mouse_pos):
                        slider_dragging = True
                        time_multiplier = max(1, min(20, (mouse_pos[0] - slider_rect.x) / 10 + 1))
                    elif spawn_star_mode:
                        all_bodies = create_new_star(mouse_pos[0], mouse_pos[1], all_bodies, zoom, center_x, center_y)
                        initial_energy = calculate_total_energy(all_bodies)
                        spawn_star_mode = False
                    else:
                        clicked_body = None
                        for body in all_bodies:
                            body_x = int(center_x + body.x * SCALE * zoom)
                            body_y = int(center_y + body.y * SCALE * zoom)
                            body_radius = max(5, int(body.radius * SCALE * zoom))
                            if math.sqrt((mouse_pos[0] - body_x)**2 + (mouse_pos[1] - body_y)**2) < body_radius:
                                clicked_body = body
                                break
                        
                        if clicked_body:
                            editing_body = clicked_body
                            selected_body = clicked_body.name
                        else:
                            dragging = True
                            drag_start = mouse_pos
                elif event.button == 3:  # Right mouse button
                    dragging = True
                    drag_start = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    slider_dragging = False
                if event.button in (1, 3):  # Left or right mouse button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if slider_dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    time_multiplier = max(1, min(20, (mouse_pos[0] - slider_rect.x) / 10 + 1))
                elif dragging:
                    mouse_pos = event.pos
                    dx = mouse_pos[0] - drag_start[0]
                    dy = mouse_pos[1] - drag_start[1]
                    center_x += dx
                    center_y += dy
                    drag_start = mouse_pos
            elif event.type == pygame.KEYDOWN:
                if editing_body:
                    if event.key == pygame.K_UP:
                        editing_body.mass *= 1.1
                    elif event.key == pygame.K_DOWN:
                        editing_body.mass /= 1.1
                    elif event.key == pygame.K_RIGHT:
                        editing_body.radius *= 1.1
                    elif event.key == pygame.K_LEFT:
                        editing_body.radius /= 1.1
                    elif event.key == pygame.K_ESCAPE:
                        editing_body = None
                    initial_energy = calculate_total_energy(all_bodies)

        # Simulation logic
        TIME_STEP = BASE_TIME_STEP * time_multiplier
        num_substeps = min(MAX_SUBSTEPS, max(1, int(TIME_STEP / 3600)))
        substep = TIME_STEP / num_substeps

        for _ in range(num_substeps):
            initial_energy, all_bodies = rk4_step(all_bodies, substep, initial_energy)
            all_bodies = check_collisions(all_bodies)
        simulation_time += TIME_STEP

        # Drawing
        screen.fill(BLACK)

        if editing_body:
            center_body = editing_body
        else:
            center_body = next(body for body in all_bodies if body.name == selected_body)
        
        view_center_x, view_center_y = center_x - center_body.x * SCALE * zoom, center_y - center_body.y * SCALE * zoom

        for star in stars:
            x, y = star
            x = (x - center_x * 0.1) % WIDTH
            y = (y - center_y * 0.1) % HEIGHT
            pygame.draw.circle(screen, WHITE, (int(x), int(y)), 1)

        for body in all_bodies:
            body.draw(screen, view_center_x, view_center_y, zoom, font)

        restart_button, reset_zoom_button, trail_button, spawn_star_button, dropdown_rect, slider_rect = draw_ui(
            screen, font, selected_body, editing_body, current_trail_option, simulation_time, time_multiplier, all_bodies, dropdown_open, view_center_x, view_center_y, zoom, spawn_star_mode)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
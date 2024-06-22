# LLM Solar System Simulator

This is a little solar system simulation based on pygame. 
It was 100% developed by a large language model (LLM), more precisely the recently released (Claude 3.5 Sonnet)[https://www.anthropic.com/news/claude-3-5-sonnet] by Anthropic AI.
I did create it to test as to what complex of a program a LLM can create fully on its on, just by me prompting it for the features i would like the program to have, and fixing issues by running it and telling the LLM what the issues are.
All in all, i was seriously impressed by the capabilities of the LLM.

## Features
- Includes the Sun, planets, and major moons of the solar system.
- Full N-body simulation
- Uses the Runge-Kutta 4th order (RK4) method for accurate orbital calculations.
- Implements adaptive time-stepping to balance accuracy and performance.
- Includes collision detection and merging of celestial bodies.
- Monitors total system energy and applies corrections to maintain conservation.
- Implements a dynamic scaling system to display bodies of vastly different sizes.
- Uses a zoom feature to allow observation at different scales.
- Interactive UI: Clickable buttons for restarting simulation, resetting zoom, and changing trail options, slider for adjusting simulation speed, dropdown menu for focus selection.
- Ability to click on bodies to edit their properties (mass and radius).
- Real-time updates of the simulation based on edited properties.
- Feature to spawn new stars at user-specified locations.

## Known issues
- At high simulation speeds, the orbits of some moons seem to be unstable.

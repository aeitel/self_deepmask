require 'TrainPlotter'
local plotter = TrainPlotter.new('out.json')

plotter:info({created_time=io.popen('date'):read(),
              tag='My First Plot'})

plotter:add('Accracy', 'Train', 1, 0.5)
plotter:add('Accracy', 'Train', 2, 0.7)
plotter:add('Accracy', 'Train', 3, 0.8)
plotter:add('Accracy', 'Test', 1, 0.45)
plotter:add('Accracy', 'Test', 2, 0.6)
plotter:add('Accracy', 'Test', 3, 0.7)

plotter:add('Loss', 'Train', 1, 0.5)
plotter:add('Loss', 'Train', 2, 0.7)
plotter:add('Loss', 'Train', 3, 0.8)
plotter:add('Loss', 'Test', 1, 0.45)
plotter:add('Loss', 'Test', 2, 0.6)
plotter:add('Loss', 'Test', 3, 0.7)



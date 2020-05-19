import json

class TrainPlotter:

    def __init__(self, path):
        self.path = path
        self.figures = {}

    def info(self, s): 
        self.figures['info_str'] = s 

    def add(self, fig_id, plot_id, it, data):
        if data != data:
            data = -1

        if fig_id not in self.figures:
            self.figures[fig_id] = {}
            self.figures[fig_id]['data'] = []
            self.figures[fig_id]['layout'] = {'title':fig_id}

        fig_data = self.figures[fig_id]['data']
        plot = None
        for v in fig_data:
            if v['name'] == plot_id:
                plot = v 
     
        if plot is None:
            plot = {'name':plot_id, 'x':[], 'y':[]}
            fig_data.append(plot)

        plot['x'].append(it)
        plot['y'].append(data)

        if self.path:
            with open(self.path, 'w') as fout:
                json.dump(self.figures, fout)

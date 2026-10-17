import time


class TimeoftheDayColormap():
    def timeofthedaycolormap(self):
        t = time.localtime(time.time())
        if t.tm_hour >= 0 and t.tm_hour < 4:
            return "gist_earth"
        if t.tm_hour >= 4 and t.tm_hour < 8:
            return "terrain"
        if t.tm_hour >= 8 and t.tm_hour < 12:
            return "gist_stern"
        if t.tm_hour >= 12 and t.tm_hour < 16:
            return "nipy_spectral"
        if t.tm_hour >= 16 and t.tm_hour < 20:
            return "cubehelix"
        if t.tm_hour >= 20 and t.tm_hour < 24:
            return "gnuplot"
        # ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
        #  'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
        #  'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
        #  'turbo', 'nipy_spectral', 'gist_ncar'])
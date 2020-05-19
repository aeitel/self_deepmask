package = "TrainPlot"
version = "0.1-1"
source = {
  url = "git://github.com/joeyhng/trainplot"
}
dependencies = {
  "lua"
}
build = {
  type = "builtin",
  modules = {
    ['TrainPlotter'] = 'trainplot.lua'
  }
}

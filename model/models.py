from .BNUDC import Model

def create_model(opt):
	model = Model()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model

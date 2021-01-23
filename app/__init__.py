from flask_restplus import Api
from flask import Blueprint

from .main.controller.user_controller import api as user_ns
from .main.controller.auth_controller import api as auth_ns
from .main.controller.maintenance.regression_controller import api as regression_ns
# from .main.controller.maintenance.classification_controller import api as classification_ns
from .main.controller.maintenance.simulation_controller import api as simulation_ns

blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='FLASK RESTPLUS API FOR PREDICTIVE MAINTENANCE',
          version='1.0',
          description='Flask restplus web service for predictive maintenance'
          )

api.add_namespace(user_ns, path='/user')
api.add_namespace(auth_ns)
api.add_namespace(regression_ns)
# api.add_namespace(classification_ns)
api.add_namespace(simulation_ns)
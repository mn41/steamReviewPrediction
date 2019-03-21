#
# Server for REST endpoints
#
# run as: python server.py
#
# curl -X POST http://localhost:5000/recommend -H 'Content-Type: application/json' -d '{"review":"I like video game","game":"dota"}'
#

from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('review')
parser.add_argument('game')

class RecommendResource(Resource):
    #   
    # Sample function, takes arguments pulled from HTTP POST body
    #
    def _func(self, review, game):
        return {
            'review': review,
            'game': game,
            'recommend': True
        }

    #
    # Handle HTTP POST
    #
    def post(self):
        args = parser.parse_args()
        result = self._func(args['review'], args['game'])
        return result, 200

api.add_resource(RecommendResource, '/recommend')

if __name__ == '__main__':
    app.run(debug=True)


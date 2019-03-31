#
# Server for REST endpoints
#
# run as: python server.py
#
# curl -X POST http://localhost:5000/recommend -H 'Content-Type: application/json' -d '{"review":"I like video game","game":"dota"}'
#

from flask import Flask
from flask_restful import Resource, Api, reqparse, abort

from RNNSteam import make_rnn

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('review')
parser.add_argument('game')

#
# Map each game here
#
game_map = (
    'batmanarkhamnight',
    'batmanarkhamnightsplit',
    'dota2',
    'dota2split',
    'gtaV',
    'gtaVsplit',
    'nomanssky',
    'nomansskysplit',
    'payday2',
    'payday2split',
)


class RecommendResource(Resource):
    """   
    API Resource handler for recommend API
    """
    def _func(self, review, game):
        """
        Drive calling the scoring engine
        """
        try:
            training_percent = 10
            reviews_txt_file = '{0}reviews.txt'.format(game)
            scores_txt_file = '{0}scores.txt'.format(game)
            saved_name = '{0}.saved'.format(game)

            mean = make_rnn(reviews_txt_file, scores_txt_file, training_percent, saved_name)
            return {
                'review': review,
                'game': game,
                'mean': mean,
                'recommend': True
            }
        except Exception as ex:
            self._abort('Exception: {0}'.format(ex), 400)
    
    def _abort(self, msg, status):
        """
        Abort on error
        """
        abort(status, message=msg)

    def post(self):
        """
        Handle HTTP POST
        """

        #
        # Validate payload
        #
        args = parser.parse_args()
        if args['review'] is None:
            self._abort('Must supply "review" in post data', 404)
        elif args['game'] is None:
            self._abort('Must supply "game" in post data', 404)
        elif args['game'] not in game_map:
            self._abort('Game "{0}" not in game map'.format(args['game']), 404)

        result = self._func(args['review'], args['game'])
        return result, 200

api.add_resource(RecommendResource, '/recommend')

if __name__ == '__main__':
    app.run(debug=True)


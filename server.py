#
# Server for REST endpoints
#
# run as: python server.py
#
# curl -X POST http://localhost:5000/recommend -H 'Content-Type: application/json' -d '{"review":"I like video game","game":"dota"}'
#

from flask import Flask
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse, abort

from RestModelTest import make_rnn

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('review')
parser.add_argument('score')
parser.add_argument('game')

#
# Map each game here
#
game_map = (
    'batman',
    'batmanarkhamknightsplit',
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
    def _func(self, review, score, game):
        """
        Drive calling the scoring engine
        """
        try:
            reviews_txt_file = '{0}reviews.txt'.format(game)
            scores_txt_file = '{0}scores.txt'.format(game)

            isRecommended = make_rnn(reviews_txt_file, scores_txt_file, review, score, game)
            return {
                'review': review,
                'game': game,
                'recommended': isRecommended
            }
        except Exception as ex:
            print(ex)
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
        elif args['score'] is None:
            self._abort('Must supply "game" in post data', 404)
        elif args['game'] is None:
            self._abort('Must supply "game" in post data', 404)
        elif args['game'] not in game_map:
            self._abort('Game "{0}" not in game map'.format(args['game']), 404)

        result = self._func(args['review'], args['score'], args['game'])
        return result, 200

api.add_resource(RecommendResource, '/recommend')

if __name__ == '__main__':
    app.run(debug=True)


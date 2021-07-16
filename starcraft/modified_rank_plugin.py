class ModifiedRank():

    name = "ModifiedRank"
                    
    def handleEndGame(self, event, replay):
        if "spawningtool" in replay.filename:
            replay.players[0].highest_league = 7
            replay.players[1].highest_league = 7

class ActiveSelection():

    name = "ActiveSelection"
                    
    def handleCommandEvent(self, event, replay):
        event.active_selection = event.player.selection[10]

    def handleControlGroupEvent(self, event, replay):
        event.active_selection = event.player.selection[10]

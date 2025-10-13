import json
import random
final = []
with open('raw_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        input = item.get("input", "")
        chat = input.get("chat")
        profile = input.get("profile", "")
        traits = input.get("traits", "")
        personality = input.get("personality", "")
        world = input.get("world", "")
        output = item.get("output", "")
        profileSelection = output.get("profile").get("selection", [])
        personalitySelection = output.get("personality").get("selection", [])
        traitsSelection = output.get("traits").get("selection", [])
        worldSelection = output.get("world").get("selection", [])

        selectedProfile = [profile[i] for i in profileSelection if i < len(profile)] if profileSelection else []
        selectedPersonality = [personality[i] for i in personalitySelection if i < len(personality)] if personalitySelection else []
        selectedTraits = [traits[i] for i in traitsSelection if i < len(traits)] if traitsSelection else []
        selectedWorld = [world[i] for i in worldSelection if i < len(world)] if worldSelection else []
        
        concatinput = profile + traits + personality + world
        concatSelection = selectedProfile + selectedTraits + selectedPersonality + selectedWorld
        random.shuffle(concatinput)
        selectIdxAfterShuffle = [concatinput.index(i) for i in concatSelection if i in concatinput]
        selectIdxAfterShuffle.sort()
        finalInput = [str(i)+"."+concatinput[i] for i in range(len(concatinput))]
        if len(finalInput)>0:
            result = {
                "input": chat+"\nHere starts memorys:\n"+"\n".join(finalInput),
                "output": str(selectIdxAfterShuffle)
            }
            
            final.append(result)

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
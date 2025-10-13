import re
import ast
separator = ["profile:", "personality:", "traits:", "world:", "background:","next_conversation:"]
final = []
with open('initial.md', 'r', encoding='utf-8') as f:
    for line in f:
        result = {
            "input":{
                "chat": [],
                "profile": [],
                "personality": [],
                "traits": [],
                "world": [],
                "background": [],
            },
            "output": {
                "profile": {},
                "personality": {},
                "traits": {},
                "world": {},
                "background": {},
            }
        }
        try:
            input, output = line.split("message_type: user",1)
        except:
            print("Error: line does not contain 'message_type: user', skip")
            continue

        first_sep_pos = -1
        for s in separator:
            pos = input.find(s)
            if pos != -1 and (first_sep_pos == -1 or pos < first_sep_pos):
                first_sep_pos = pos
        
        if first_sep_pos != -1:
            result["input"]["chat"] = input[:first_sep_pos].strip().split("character_name:")[1]
        else:
            result["input"]["chat"] = input.strip().split("character_name:")[1]
        result["input"]["chat"] = "character_name: " + result["input"]["chat"].strip()
        for key in separator:
            # Process input
            start_index = input.find(key)
            if start_index != -1:
                start_index += len(key)
                end_index = -1
                for next_key in separator:
                    if next_key != key:
                        found_index = input.find(next_key, start_index)
                        if found_index != -1:
                            if end_index == -1 or found_index < end_index:
                                end_index = found_index
                
                if end_index != -1:
                    value = input[start_index:end_index].strip()
                else:
                    value = input[start_index:].strip()
                
                if value:
                    items = re.split(r'\d+\.\s*', value)
                    items = [item.strip() for item in items if item.strip()]
                    result["input"][key.replace(":", "")] = items
                else:
                    result["input"][key.replace(":", "")] = "never reach here"

            # Process output
            start_index = output.find(key)
            if start_index != -1:
                start_index += len(key)
                end_index = -1
                for next_key in separator:
                    if next_key != key:
                        found_index = output.find(next_key, start_index)
                        if found_index != -1:
                            if end_index == -1 or found_index < end_index:
                                end_index = found_index

                if end_index != -1:
                    value = output[start_index:end_index].strip()
                else:
                    value = output[start_index:].strip()

                # New logic to extract activated_memory
                match = re.search(r'<activated_memory>(\[.*?\])</activated_memory>', value)
                key = key.replace(":", "")
                result["output"][key] = {
                    "think":value,
                    "selection": None
                }
                if match:
                    try:
                        selection_list = ast.literal_eval(match.group(1))
                        result["output"][key]["selection"] = selection_list
                    except (ValueError, SyntaxError):
                        pass
        
        final.append(result)

import json
with open('raw_data.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
        
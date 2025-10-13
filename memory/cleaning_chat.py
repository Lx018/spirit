import re
import ast
final = []
with open('chat.md', 'r', encoding='utf-8') as f:
    for line in f:
        line_dict = {}
        # Only divide by parts containing 'memory' or 'chat_history'
        parts = re.split(r'(\b\w*(?:memory|chat_history|message_type|result|next_conversation)\w*:)', line)

        if parts[0].strip():
            line_dict['content_before'] = parts[0].strip()

        # The rest are key-value pairs
        for i in range(1, len(parts), 2):
            key = parts[i].replace(":", "").strip()
            value = ""
            if i + 1 < len(parts):
                value = parts[i+1].strip()
            
            if key: # Make sure key is not empty
                if "memory" in key:
                    items = re.split(r'\d+\.', value)
                    items = [item.strip() for item in items if item.strip()]
                    line_dict[key] = items
                else:
                    if "result" not in key:
                        line_dict[key] = value
        
        # Merge all memory keys into a single 'memory' key
        merged_memory_list = []
        keys_to_delete = []
        for key, value in line_dict.items():
            if "memory" in key:
                assert isinstance(value, list), f"Value for key '{key}' is not a list"
                merged_memory_list.extend(value)
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del line_dict[key]

        if merged_memory_list:
            line_dict['memory'] = merged_memory_list

        if line_dict:
            final.append(line_dict)


import json
with open('raw_chat_data.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
        
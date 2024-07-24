import pickle

file_path = './processed/speechpref_train.pkl'

data_train = {
    0: [
        {
            "prompt": "How old is the Greek language?",
            "output": "The Greek language is over 3500 years old and it is one of the oldest documented languages in the Western world. Would you like to learn more about the history of the Greek language?"
        },
        {
            "prompt": "Give me 3 different things that a smartphone can do better than a landline phone.",
            "output": "Smartphones can do many things that landline phones can't. They can access the internet, send text messages, and run apps. Would you like to learn more about other differences between smartphones and landline phones?"
        },
        {
            "prompt": "What are some benefits in meditating?",
            "output": "Meditation can help reduce stress, improve focus and concentration, promote emotional well-being, and enhance self-awareness. Do you have any questions about these benefits?"
        },
        {
            "prompt": "What to ask when interviewing doulas?",
            "output": "When interviewing doulas, you might want to ask about their training and certifications, how many births they've attended as a doula, and their philosophy and approach to supporting birthing families. You should also ask about how they handle unexpected situations, their availability around your due date, and their fee structure and the services they provide. You may also ask for references or testimonials from previous clients. Remember, it's important to find a doula who aligns with your values and makes you feel comfortable and supported during this important time."
        },
        {
            "prompt": "Is it safe to invest in cryptocurrency?",
            "output": "It depends on the specific cryptocurrency you are considering investing in. Cryptocurrencies are volatile, and there is significant risk involved. It's important to do your own research and carefully consider your options before making any investment decisions. I can't provide investment advice but I can share more information about specific cryptocurrencies if you would like."
        },
        {
            "prompt": "Why is pickleball so popular in the US right now?",
            "output": "Pickleball is popular in the US because it's a fun, inclusive and easy-to-learn sport. It's lower impact than sports like tennis so it appeals to a broad range of ages. Plus, you don't need a ton of gear to play, which makes it accessible. There's just something about hitting that whiffle ball that's super satisfying! Would you like to learn more about how it became popular?"
        },
        {
            "prompt": "What are five capital cities in Asia?",
            "output": "Some of the capital cities in Asia include Tokyo, Japan; Beijing, China; New Delhi, India; Seoul, South Korea; and Bangkok, Thailand. Would you like to learn more about these cities or other capital cities in Asia?"
        }
    ]
}

data_test = {
    0: [
        {
            "prompt": "What are some benefits of learning a new language?",
            "output": "Learning a new language can improve cognitive abilities like memory, focus, problem solving skills, and creativity. It can even delay the onset of age-related cognitive decline. Learning another language also helps you gain cultural awareness, allowing you to connect with more people and appreciate different perspectives. It also boosts your career prospects as it opens up new career opportunities. Which of these benefits apeal to you the most?"
        },
        {
            "prompt": "What are some common misconceptions about climate change?",
            "output": "Some common misconceptions about climate change are that it's not real, that it's not caused by human activity, and that it won't have serious consequences. But climate change is a real and urgent threat that is primarily caused by human activities like burning fossil fuels and deforestation. Is there a specific misconception you'd like to learn more about?"
        },
        {
            "prompt": "How can I reduce food waste at home?",
            "output": "There are many ways to reduce food waste at home, such as planning your meals, storing food properly, and using up leftovers. You can also compost food scraps and support businesses that are committed to reducing food waste. Which of these strategies are you most interested in learning more about?"
        },
    ]
}


# Writing the data to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(data_train, file)
    
with open(file_path.replace("train", "test"), 'wb') as file:
    pickle.dump(data_test, file)
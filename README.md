
# RecurrentJava

RecurrentJava is a reimplementation of Andrej Karpathy's [RecurrentJS](https://github.com/karpathy/recurrentjs), in Java.

It currently features:

- Deep Recurrent Neural Networks
- Long Short-Term Memory Networks
- Gated Recurrent Unit Neural Networks
- Backpropagation Through Time handled via Automatic Differentiation.

ExamplePaulGraham.java shows how to do character-by-character sentence prediction and generation.

Sample output:

```
========================================
REPORT:

calculating perplexity over entire data set...

Median Perplexity = 1.4959

Temperature 1.0 prediction:
	"there's a more kemmaces of meanness that hade? tagh o; mool"
	"it fart dect about twish i could see gve..."

Temperature 0.75 prediction:
	"that's not absolutely note a lot of the startup? path they'll should owt"
	"i realize how crazy all thi..."

Temperature 0.5 prediction:
	"the most stripiess to more here that happens never get them"
	"if you do that role kropate that's the w..."

Temperature 0.25 prediction:
	"the person who needs something making the same spignf befart"
	"the startup founders who never about wh..."

Temperature 0.1 prediction:
	"the startup founders who never about which in your expanding, it's a sign when idea way we don't the..."

Argmax prediction:
	"the problem is not that most towns kill startups"
	"the problem is not that most towns kill startups"
	"th..."
========================================
```

## License
MIT
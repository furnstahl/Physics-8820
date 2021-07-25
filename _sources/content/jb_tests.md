---
substitutions:
  key1: "I'm a **substitution**"
  key2: |
    ```{note}
    {{ key1 }}
    ```
  key3: |
    ```{image} /_images/fun-fish.png
    :alt: fishy
    :width: 200px
    ```
  key4: example
---

# Tests of Jupyter Book features

Click the `.md` button under the download icon on the top middle right to see the markdown code for this section. The file theme.css has color assignments and the like. Customize? (See https://jupyterbook.org/advanced/html.html#custom-assets)


## colon_fence

To allow you to use ::: fences for admonitions, in order to make them easier to render in interfaces that do not support MyST.

::::{important}
:::{note}
This text is **standard** _Markdown_. We're using the "colon_fence" extension.
:::
::::

## replacements

Test of "replacements" extension: 
(r), (R)  en-dash: -- and em-dash: ---

Trying "linkify" extension: google.com

## substitutions

Inline: {{ key1 }}

Block level:

{{ key2 }}

| col1     | col2     |
| -------- | -------- |
| {{key2}} | {{key3}} |

## definition lists

Term 1
: Definition

Term 2
: Definition

## images

```{image} /_images/fun-fish.png
:alt: fishy
:class: bg-primary
:width: 200px
:align: center
```

Now for some text in between.

:::{figure-md} fig-target
:class: myclass

<img src="/_images/fun-fish.png" alt="fishy" class="bg-primary mb-1" width="200px">

This is a caption in **Markdown**
:::


[Go to the fish!](fig-target)


## HTML admonitions

<div class="admonition note" name="html-admonition" style="background: lightgreen; padding: 10px">
<p class="title">This is the **title**</p>
This is the *content* of an admonition.
</div>

<div class="admonition">
<p>Some **content**</p>
  <div class="admonition tip">
  <div class="title">A *title*</div>
  <p>Paragraph 1</p>
  <p>Paragraph 2</p>
  </div>
</div>

### Customized admonitions

:::{admonition} Extra credit
An "extra credit" exercise is presented here.
:::

## Dropdown answer

The old hidden fish trick!
```{toggle}
Some hidden toggle content!

![](/_images/fun-fish.png)
```



Maybe this needs to be in an .ipynb file?

```{admonition} Question
The matrix product of 2 rank 2 tensors
```
```{admonition} Answer 
:class: dropdown 
  First answer is: $ a_{ij} b_{jk} = c_{ik} $
```

Let's try again here.

```{admonition} Answer
:class: dropdown
\begin{equation}
  \mbox{Second answer is:}\quad  a_{ij} b_{j} = c_{i}
\end{equation}
```

Spread things out a bit.

```{admonition} Click the button to reveal!
:class: dropdown
Some hidden toggle content!

![](/_images/8820_icon.png)
```


Different admonitions: note, warning, tip, caution, attention, danger, error, hint, important


## Panels

````{panels}
Content of the left panel.

{badge}`example-badge,badge-primary`

---

```{link-button} /content/zbibliography
:text: Clickable right panel
:type: ref
:classes: stretched-link
```

````


Controlling the look and feel (here the  border):

````{panels}
:column: col-4
:card: border-2
Header A
^^^
Body A
---
Header B
^^^
Body B
---
Header C
^^^
Body C
````


## More dropdowns

```{dropdown} Here's my dropdown
And here's my dropdown content
```

Turning an admonition into a dropdown (see above as well):

```{note}
:class: dropdown
The note body will be hidden!
```


    ::::{admonition} Question
    What is the pdf $p(x)$ if we know **definitely** that $x = x_0$ (i.e., fixed)?
    :::{admonition} Answer 
    :class: dropdown 
    $p(x) = \delta(x-x_0)\quad$  [Note that $p(x)$ is normalized.]
    :::
    ::::



## Footnotes

You can include footnotes in your book using standard Markdown syntax. This will include a numbered reference to the footnote in-line, and append the footnote to a list of footnotes at the bottom of the page.

To create a footnote, first insert a reference in-line with this syntax (look at the markdown for how this plays out): [^mylabel]. 

[^mylabel]: My footnote text.

You can define [^mylabel] anywhere in the page, though its definition will always be placed at the bottom of your built page.


## Quotations and epigraphs

Standard markdown first:

> Here is a cool quotation.
>
> From me, Jo the Jovyan

Now with the epigraph admonition. (Epigraphs draw more attention to a quote and highlight its author. You should keep these relatively short so that they don’t take up too much vertical space.)


```{epigraph}
Here is a cool quotation.

-- Jo the Jovyan
```

## Glossaries

```{glossary}
Term one
  An indented explanation of term 1

A second term
  An indented explanation of term2
```

To reference terms in your glossary, use the {term} role. For example, {term}`Term one` becomes Term one and {term}`A second term` becomes A second term.


## Tabbed content

```{tabbed} Tab 1 title
My first tab
```

```{tabbed} Tab 2 title
My second tab with `some code`!
```

## Margin and sidebar content

```{margin} **My margin title**
Here is my margin content, it is pretty cool!
```
You can specify content that should exist in the right margin. 
This will behave like a regular sidebar until the screen hits a certain width, at which point this content will “pop out” to the right white space. 
Put the margin syntax at the start of the paragraph you want to add margin content to.


```{sidebar} **My sidebar title**
Here is my sidebar content, it is pretty cool!
```
Content sidebars exist in-line with your text, but allow the rest of the page to flow around them, rather than moving to the right margin.
As with the margin notes, put the sidebar syntax at the start of the paragraph that will wrap around the sidebar.
Note how the content wraps around the sidebar to the right. However, the sidebar text will still be in line with your content. There are certain kinds of elements, such as “note” blocks and code cells, that may clash with your sidebar. If this happens, try using a {margin} instead. 
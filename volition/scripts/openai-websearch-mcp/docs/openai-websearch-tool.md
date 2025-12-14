
# Web search
This tool searches the web for relevant results to use in a response. Learn more about the web search tool.


## properties
`type` string

> Required
> The type of the web search tool. One of:
> web_search_preview
> web_search_preview_2025_03_11


`search_context_size` string

> Optional
> Defaults to medium
> High level guidance for the amount of context window space to use for the search. One of low, medium, or high. medium is the default.

`user_location` object or null

> Optional
> Approximate location parameters > for the search.


properties of `user_location`   
`type` string

> Required
> The type of location > approximation. Always approximate.

`city` string

> Optional
> Free text input for the city of the user, e.g. San Francisco.

`country` string

> Optional
> The two-letter ISO country code of the user, e.g. US.

`region` string

> Optional
> Free text input for the region of the user, e.g. California.

`timezone` string

> Optional
> The IANA timezone of the user, e.g. America/Los_Angeles.
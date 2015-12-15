# til-france documentation

> Note: this documentation is being written.

til-france is the [Institut des Politiques Publique](www.ipp.eu)'s microsimulation software.

It is published under the [GNU Affero General Public License version 3.0](http://www.gnu.org/licenses/agpl-3.0.html).

## About this documentation

This documentation is built with the excellent [GitBook](https://github.com/GitbookIO/gitbook) tool
(see [GitBook documentation](http://help.gitbook.com/)).

It is written in [Markdown](http://help.gitbook.com/format/markdown.html)
and the source is hosted on this GitHub repository:
[TaxIPP-Life/til-france](https://github.com:TaxIPP-Life/til-france.git).

### Collaborative editing

Everybody can participate to the redaction of the documentation.

On each page is a link named "Edit this page".
Just click on it and you'll jump on GitHub on the Markdown source file of the page.
Then edit the file as explained on this GitHub documentation page:
[editing-files-in-another-user-s-repository](https://help.github.com/articles/editing-files-in-another-user-s-repository/).

Then save the file and create a [pull request](https://help.github.com/articles/creating-a-pull-request/) which will be
accepted if relevant.

### Build it yourself

If you'd like to build it by yourself, here are the steps.

```
git clone git@github.com:TaxIPP-Life/til-france.git
npm install
```

Then you can either build the documentation or launch a local HTTP server with watch mode:

```
npm run build
or
npm run watch
```

> With watch mode, open http://localhost:2050/ in your browser once the first build is done.

### Deploy (for maintainers)

To deploy the built documentation
(you must be authorized to push to [TaxIPP-Life/til-france](https://github.com/TaxIPP-Life/til-france)):

```
npm run publish
```

Then on the server, the first time:

```bash
git clone https://github.com/TaxIPP-Life/til-france.git --branch gitbook-static til-france-gitbook-static
```

The next times:

```bash
cd til-france-gitbook-static
git fetch
git reset --hard origin/til-france-static
```
